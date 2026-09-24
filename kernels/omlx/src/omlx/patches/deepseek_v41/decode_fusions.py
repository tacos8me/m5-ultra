# Portions adapted from jundot/omlx PR #3574 (MIT), hyper_connection.py.
# Projection scheduling and router are specific to the ds41 M5 worktree.
import os
from functools import cache
import mlx.core as mx

# Bitwise-identical rewrites of the decode kernels below (0 restores originals).
DS41_DECODE_KERNELS_V2 = os.environ.get("DS41_DECODE_KERNELS_V2", "1") == "1"


@cache
def _project():
    return mx.fast.metal_kernel(
        name='ds41_hc_project', input_names=['x', 'weight', 'eps'], output_names=['mix'],
        source=r'''
        uint row = threadgroup_position_in_grid.x;
        uint o = threadgroup_position_in_grid.y;
        uint t = thread_position_in_threadgroup.x;
        uint lane = t % 32, sg = t / 32;
        threadgroup float dot[8], sq[8];
        float a = 0, b = 0;
        for (uint k=t; k<D; k+=256) {
            float v = float(x[row*D+k]);
            a = fma(v, float(weight[o*D+k]), a);
            b = fma(v, v, b);
        }
        a = simd_sum(a); b = simd_sum(b);
        if (lane==0) {dot[sg]=a; sq[sg]=b;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg==0) {
            float v = simd_sum(lane<8 ? dot[lane] : 0.0f);
            float s = simd_sum(lane<8 ? sq[lane] : 0.0f);
            if (lane==0) mix[row*24+o] = v * rsqrt(s/float(D)+eps[0]);
        }
        ''')


@cache
def _project_pipelined():
    # Same per-thread FMA chains and reductions as _project; each thread issues
    # U strided loads before its FMAs so the 24 thin threadgroups are not
    # bound by one DRAM round trip per step.
    return mx.fast.metal_kernel(
        name='ds41_hc_project_pipelined', input_names=['x', 'weight', 'eps'], output_names=['mix'],
        source=r'''
        uint row = threadgroup_position_in_grid.x;
        uint o = threadgroup_position_in_grid.y;
        uint t = thread_position_in_threadgroup.x;
        uint lane = t % 32, sg = t / 32;
        threadgroup float dot[8], sq[8];
        constexpr uint STEPS = (D + 255) / 256;
        constexpr uint U = 8;
        float a = 0, b = 0;
        for (uint s0 = 0; s0 < STEPS; s0 += U) {
            float xv[U], wv[U];
            for (uint u = 0; u < U; ++u) {
                uint k = t + 256 * (s0 + u);
                bool ok = s0 + u < STEPS && k < D;
                xv[u] = ok ? float(x[row*D+k]) : 0.0f;
                wv[u] = ok ? float(weight[o*D+k]) : 0.0f;
            }
            for (uint u = 0; u < U; ++u) {
                if (s0 + u < STEPS && t + 256 * (s0 + u) < D) {
                    a = fma(xv[u], wv[u], a);
                    b = fma(xv[u], xv[u], b);
                }
            }
        }
        a = simd_sum(a); b = simd_sum(b);
        if (lane==0) {dot[sg]=a; sq[sg]=b;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg==0) {
            float v = simd_sum(lane<8 ? dot[lane] : 0.0f);
            float s = simd_sum(lane<8 ? sq[lane] : 0.0f);
            if (lane==0) mix[row*24+o] = v * rsqrt(s/float(D)+eps[0]);
        }
        ''')


def hc_project(x, weight, eps):
    kernel = _project_pipelined() if DS41_DECODE_KERNELS_V2 else _project()
    return kernel(inputs=[x, weight, mx.array([eps], mx.float32)],
        template=[('D',x.shape[-1]*4)], grid=(x.size//(x.shape[-1]*4)*256,24,1),
        threadgroup=(256,1,1), output_shapes=[(*x.shape[:-2],24)], output_dtypes=[mx.float32])[0]


@cache
def _pre():
    return mx.fast.metal_kernel(name='ds41_hc_pre_norm',
        input_names=['x','pre','weight','eps'], output_names=['y'],
        header='#pragma clang fp contract(off)\n', source=r'''
        uint row=threadgroup_position_in_grid.x, t=thread_position_in_threadgroup.x;
        uint lane=t%32, sg=t/32;
        threadgroup float sums[8];
        float v[(D+255)/256], total=0;
        for (uint j=0; j<(D+255)/256; ++j) {
            uint d=t+256*j; float z=0;
            if (d<D) {
                for (uint h=0;h<4;++h) z=z+float(x[(row*4+h)*D+d])*pre[row*4+h];
                z=float(T(z));
            }
            v[j]=z; total=total+z*z;
        }
        total=simd_sum(total);
        if(lane==0) sums[sg]=total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(sg==0) {
            float s=simd_sum(lane<8?sums[lane]:0.0f);
            if(lane==0) sums[0]=rsqrt(s/float(D)+eps[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=0;j<(D+255)/256;++j) {
            uint d=t+256*j;
            if(d<D) y[row*D+d]=T((v[j]*sums[0])*float(weight[d]));
        }
        ''')


def hc_pre_norm(x, pre, weight, eps):
    d=x.shape[-1]
    return _pre()(inputs=[x,pre,weight,mx.array([eps],mx.float32)],template=[('D',d),('T',x.dtype)],
        grid=(x.size//(4*d)*256,1,1),threadgroup=(256,1,1),
        output_shapes=[(*x.shape[:-2],d)],output_dtypes=[x.dtype])[0]


@cache
def _post():
    return mx.fast.metal_kernel(name='ds41_hc_post', input_names=['x','res','post','comb'],output_names=['y'],
        header='#pragma clang fp contract(off)\n',source=r'''
        uint z=thread_position_in_grid.x;
        uint size = D;
        for (uint dim = 0; dim + 1 < x_ndim; ++dim) size *= x_shape[dim];
        if (z >= size) return;
        uint r=z/D,d=z%D;
        for(uint j=0;j<4;++j) {
            float s=0;
            for(uint i=0;i<4;++i) s=fma(comb[r*16+i*4+j],float(res[(r*4+i)*D+d]),s);
            y[(r*4+j)*D+d]=T(post[r*4+j]*float(x[z])+s);
        }
        ''')


def hc_post(x,res,post,comb):
    return _post()(inputs=[x,res,post,comb],template=[('D',x.shape[-1]),('T',x.dtype)],
        grid=(x.size,1,1),threadgroup=(256,1,1),output_shapes=[res.shape],output_dtypes=[x.dtype])[0]


@cache
def _router():
    return mx.fast.metal_kernel(name='ds41_router_sqrtsoftplus_top6',
        input_names=['logits','bias','scale'],output_names=['ids','scores'],source=r'''
        uint r=threadgroup_position_in_grid.x,t=thread_position_in_threadgroup.x;
        threadgroup float values[384], selected[6], candidates[384];
        threadgroup uint chosen[6], maxids[12];
        threadgroup float maxima[12];
        if(t<384) {
            float a=float(logits[r*384+t]);
            float v=sqrt(max(a,0.0f)+log1p(exp(-abs(a))));
            values[t]=v; candidates[t]=v+float(bias[t]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint k=0;k<6;++k) {
            float v=candidates[t];
            float m=simd_max(v);
            uint idx=simd_min(v==m?t:0xffffffffu);
            if(t%32==0) {maxima[t/32]=m;maxids[t/32]=idx;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(t<32) {
                float v2=t<12?maxima[t]:-INFINITY;
                float m2=simd_max(v2);
                uint idx2=simd_min(t<12 && v2==m2 ? maxids[t]:0xffffffffu);
                if(t==0) {chosen[k]=idx2;selected[k]=values[idx2];candidates[idx2]=-INFINITY;}
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if(t<6) {
            float total=0;
            for(uint k=0;k<6;++k) total+=selected[k];
            ids[r*6+t]=chosen[t]; scores[r*6+t]=selected[t]/(total+1e-20f)*scale[0];
        }
        ''')


def router(logits,bias,scale):
    shape=(*logits.shape[:-1],6)
    return _router()(inputs=[logits,bias,mx.array([scale],mx.float32)],
        grid=(logits.size//384*384,1,1),threadgroup=(384,1,1),
        output_shapes=[shape,shape],output_dtypes=[mx.uint32,mx.float32])


@cache
def _mix():
    return mx.fast.metal_kernel(
        name='ds41_mix_sinkhorn',input_names=['mix','scale','base','eps'],
        output_names=['pre','post','comb'],
        header='#pragma clang fp reassociate(off)\n#pragma clang fp contract(off)\n',
        source=r"""
        uint row=thread_position_in_grid.x;
        uint rows=1;
        for(uint dim=0;dim+1<mix_ndim;++dim)rows*=mix_shape[dim];
        if(row>=rows)return;
        float a[16];
        for(uint i=0;i<4;i++){
            float p=mix[row*24+i]*scale[0]+base[i];
            float o=mix[row*24+4+i]*scale[1]+base[4+i];
            pre[row*4+i]=1.0f/(1.0f+exp(-p))+eps[0];
            post[row*4+i]=2.0f/(1.0f+exp(-o));
            float m=-INFINITY;
            for(uint j=0;j<4;j++){
                uint k=i*4+j;
                a[k]=mix[row*24+8+k]*scale[2]+base[8+k];
                m=max(m,a[k]);
            }
            float sum=0;
            for(uint j=0;j<4;j++) {a[i*4+j]=exp(a[i*4+j]-m);sum+=a[i*4+j];}
            for(uint j=0;j<4;j++) a[i*4+j]=a[i*4+j]/sum+eps[0];
        }
        for(uint it=0;it<ITERS;it++){
            if(it>0)for(uint i=0;i<4;i++){
                float sum=0;
                for(uint j=0;j<4;j++)sum=a[i*4+j]+sum;
                sum+=eps[0];
                for(uint j=0;j<4;j++)a[i*4+j]/=sum;
            }
            for(uint j=0;j<4;j++){
                float sum=0;
                for(uint i=0;i<4;i++)sum=a[i*4+j]+sum;
                sum+=eps[0];
                for(uint i=0;i<4;i++)a[i*4+j]/=sum;
            }
        }
        for(uint i=0;i<16;i++)comb[row*16+i]=a[i];
        """)


@cache
def _mix_parallel():
    # One lane per 4x4 entry (16 lanes per row) instead of one thread per row.
    # Every lane repeats the serial kernel's exact per-row/column operation
    # order over shuffled values, so all outputs stay bitwise identical while
    # the 20 Sinkhorn iterations no longer serialize 16 divisions each.
    return mx.fast.metal_kernel(
        name='ds41_mix_sinkhorn_parallel',input_names=['mix','scale','base','eps'],
        output_names=['pre','post','comb'],
        header='#pragma clang fp reassociate(off)\n#pragma clang fp contract(off)\n',
        source=r"""
        uint gid=thread_position_in_grid.x;
        uint lane=thread_index_in_simdgroup;
        uint first=lane&~15u;
        uint row=gid/16, t=gid%16, i=t/4, j=t%4;
        uint rows=1;
        for(uint dim=0;dim+1<mix_ndim;++dim)rows*=mix_shape[dim];
        bool active=row<rows;
        uint r=active?row:0;
        if(active && t<4){
            float p=mix[r*24+t]*scale[0]+base[t];
            float o=mix[r*24+4+t]*scale[1]+base[4+t];
            pre[r*4+t]=1.0f/(1.0f+exp(-p))+eps[0];
            post[r*4+t]=2.0f/(1.0f+exp(-o));
        }
        float a=mix[r*24+8+t]*scale[2]+base[8+t];
        float m=-INFINITY;
        for(uint jj=0;jj<4;jj++) m=max(m,simd_shuffle(a,first+i*4+jj));
        a=exp(a-m);
        float sum=0;
        for(uint jj=0;jj<4;jj++) sum+=simd_shuffle(a,first+i*4+jj);
        a=a/sum+eps[0];
        for(uint it=0;it<ITERS;it++){
            if(it>0){
                float rs=0;
                for(uint jj=0;jj<4;jj++) rs=simd_shuffle(a,first+i*4+jj)+rs;
                rs+=eps[0];
                a/=rs;
            }
            float cs=0;
            for(uint ii=0;ii<4;ii++) cs=simd_shuffle(a,first+ii*4+j)+cs;
            cs+=eps[0];
            a/=cs;
        }
        if(active) comb[r*16+t]=a;
        """)


def mix_sinkhorn(mixes, scale, base, eps, iters):
    rows=mixes.size//24
    if DS41_DECODE_KERNELS_V2:
        return _mix_parallel()(inputs=[mixes,scale,base,mx.array([eps],mx.float32)],
                  template=[('ITERS',max(1,iters))],
                  grid=((rows+1)//2*32,1,1),threadgroup=(32,1,1),
                  output_shapes=[(*mixes.shape[:-1],4),(*mixes.shape[:-1],4),(*mixes.shape[:-1],4,4)],
                  output_dtypes=[mx.float32]*3)
    return _mix()(inputs=[mixes,scale,base,mx.array([eps],mx.float32)],
                  template=[('ITERS',max(1,iters))],
                  grid=(rows,1,1),threadgroup=(32,1,1),
                  output_shapes=[(*mixes.shape[:-1],4),(*mixes.shape[:-1],4),(*mixes.shape[:-1],4,4)],
                  output_dtypes=[mx.float32]*3)


# mx.compile prints float constants with 7 significant digits; the reference
# pack graph therefore clamps with this rounded 448 * 2**-126, not the exact one.
_MINIMUM_LITERAL = "%.7g" % (448.0 * 2.0**-126)


@cache
def _pack_fp8():
    # quantization.pack_activation(x) (bits=8, group 32, power-of-two scales)
    # in one pass: same MLX op semantics (NaN-propagating max/min, precise
    # log2, (x>0)-(x<0) sign, PyTorch-derived fp8 encoding), one SIMD group
    # per 32-value scale group.
    return mx.fast.metal_kernel(
        name='ds41_pack_fp8_rows', input_names=['x'], output_names=['packed'],
        header=f'''
        #define MINIMUM {_MINIMUM_LITERAL}
''' + r'''
        inline float ds41_maximum(float x, float y) { if (metal::isnan(x)) return x; return x > y ? x : y; }
        inline float ds41_minimum(float x, float y) { if (metal::isnan(x)) return x; return x < y ? x : y; }
        inline float ds41_pow2(float e) { return as_type<float>(uint32_t(e + 127.0f) << 23); }
        inline uint8_t ds41_to_fp8(float f) {
            uint32_t fp8_max = 543 << 21;
            uint32_t denorm_mask = 141 << 23;
            uint32_t f_bits = as_type<uint32_t>(f);
            uint32_t sign = f_bits & 0x80000000;
            uint8_t bits;
            f_bits ^= sign;
            if (f_bits >= fp8_max) {
                bits = 0x7E;
            } else if (f_bits < (121 << 23)) {
                f_bits = as_type<uint32_t>(as_type<float>(f_bits) + as_type<float>(denorm_mask));
                bits = static_cast<uint8_t>(f_bits - denorm_mask);
            } else {
                uint8_t mant_odd = (f_bits >> 20) & 1;
                f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
                f_bits += mant_odd;
                bits = static_cast<uint8_t>(f_bits >> 20);
            }
            bits |= static_cast<uint8_t>(sign >> 24);
            return bits;
        }
        ''',
        source=r'''
        const uint gid = thread_position_in_grid.x;
        const uint lane = thread_index_in_simdgroup;
        const uint row = gid / D, d = gid % D;
        const float f = float(x[gid]);
        const float amax = ds41_maximum(simd_max(metal::abs(f)), static_cast<float>(MINIMUM));
        const float exponent = ds41_maximum(metal::ceil(metal::precise::log2(amax / 448.0f)), -126.0f);
        const float scaled = ds41_minimum(ds41_maximum(f / ds41_pow2(exponent), -448.0f), 448.0f);
        const float a = ds41_minimum(metal::abs(scaled), 448.0f);
        const float step_exponent = metal::floor(metal::precise::log2(ds41_maximum(a, 0x1p-9f)));
        const float step = ds41_pow2(ds41_maximum(step_exponent - 3.0f, -9.0f));
        const float sign = float(int(scaled > 0.0f) - int(scaled < 0.0f));
        const float q = sign * ds41_minimum(metal::rint(a / step) * step, 448.0f);
        const uint out_row = row * (D + D / 32);
        packed[out_row + d] = ds41_to_fp8(q);
        if (lane == 0) packed[out_row + D + d / 32] = uint8_t(exponent + 127.0f);
        ''')


def pack_fp8(x):
    """Bitwise pack_activation(x) for short decode rows (bits=8, group_size=32)."""
    d = x.shape[-1]
    return _pack_fp8()(inputs=[x], template=[('D', d)], grid=(x.size, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[(*x.shape[:-1], d + d // 32)], output_dtypes=[mx.uint8])[0]


# Bitwise replica of MLX's GemvWide (k_lanes 32, one pass of 2-5 vectors),
# which mx.einsum("bsgd,grd->bsgr") dispatches for the grouped wo_a
# projection: every lane keeps its 4-wide K slots, the unroll-8 block
# order and the shuffle-down reduction, but serves two rows so each
# activation chunk it loads feeds both.
@cache
def _grouped_gemv():
    return mx.fast.metal_kernel(
        name='ds41_grouped_gemv_rows', input_names=['x', 'weight'], output_names=['y'],
        source=r"""
        const uint3 tid = threadgroup_position_in_grid;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        constexpr int R = 2;
        constexpr int unroll = 8;
        constexpr int n_v4 = K / 4;
        constexpr int n_main = n_v4 - n_v4 % (32 * unroll);
        const int g = tid.z;
        const int row0 = tid.y * (4 * R) + simd_gid * R;
        const device vec<T, 4>* w4[R];
        for (int r = 0; r < R; r++) {
            const int row = min(row0 + r, N - 1);
            w4[r] = (const device vec<T, 4>*)(weight + (size_t(g) * N + row) * K);
        }
        const device vec<T, 4>* x4[M];
        for (int v = 0; v < M; v++) x4[v] = (const device vec<T, 4>*)(x + (size_t(v) * G + g) * K);
        float result[R][M];
        for (int r = 0; r < R; r++) for (int v = 0; v < M; v++) result[r][v] = 0;
        for (int base = 0; base < n_main; base += 32 * unroll) {
            float acc[R][M];
            for (int r = 0; r < R; r++) for (int v = 0; v < M; v++) acc[r][v] = 0;
            for (int i = 0; i < unroll; i++) {
                const int idx = base + i * 32 + simd_lid;
                float4 xq[M];
                for (int v = 0; v < M; v++) xq[v] = float4(x4[v][idx]);
                for (int r = 0; r < R; r++) {
                    const float4 wf = float4(w4[r][idx]);
                    for (int v = 0; v < M; v++) acc[r][v] += dot(wf, xq[v]);
                }
            }
            for (int r = 0; r < R; r++) for (int v = 0; v < M; v++) result[r][v] += acc[r][v];
        }
        for (int idx = n_main + simd_lid; idx < n_v4; idx += 32) {
            for (int r = 0; r < R; r++) {
                const float4 wf = float4(w4[r][idx]);
                for (int v = 0; v < M; v++) result[r][v] += dot(wf, float4(x4[v][idx]));
            }
        }
        for (int r = 0; r < R; r++)
            for (int v = 0; v < M; v++)
                for (ushort off = 16; off >= 1; off >>= 1)
                    result[r][v] += simd_shuffle_down(result[r][v], off);
        if (simd_lid == 0)
            for (int r = 0; r < R; r++)
                if (row0 + r < N)
                    for (int v = 0; v < M; v++)
                        y[(size_t(v) * G + g) * N + row0 + r] = static_cast<T>(result[r][v]);
        """)


def grouped_gemv_supported(grouped, weight):
    """Shapes MLX sends to GemvWide with k_lanes 32 in a single pass."""
    return (
        DS41_DECODE_KERNELS_V2
        and grouped.ndim == 4
        and grouped.shape[0] == 1
        and 2 <= grouped.shape[1] <= 5
        and grouped.dtype in (mx.bfloat16, mx.float16)
        and weight.dtype == grouped.dtype
        and weight.ndim == 3
        and weight.shape[0] == grouped.shape[2]
        and weight.shape[2] == grouped.shape[3]
        and grouped.shape[3] % 4 == 0
        and weight.shape[1] > 64
        and mx.default_device() == mx.gpu
    )


def grouped_gemv(grouped, weight):
    """Bitwise mx.einsum("bsgd,grd->bsgr", grouped, weight) for 2-5 rows."""
    length, groups, k = grouped.shape[1:]
    n = weight.shape[1]
    return _grouped_gemv()(
        inputs=[grouped, weight],
        template=[('T', grouped.dtype), ('K', k), ('N', n), ('M', length), ('G', groups)],
        grid=(32, (n + 7) // 8 * 4, groups), threadgroup=(32, 4, 1),
        output_shapes=[(1, length, groups, n)], output_dtypes=[grouped.dtype])[0]
