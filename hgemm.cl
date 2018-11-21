#define USE_TC

void HgemmBody(const int kSizeM, const int kSizeN, const int kSizeK,
                  #if SA == 1
                    __local short* alm,
                  #endif
                  #if SB == 1
                    __local short* blm,
                  #endif
                  const __global half* restrict agm,
                  const __global half* restrict bgm,
                  __global half* restrict cgm)
{
    int laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));

    // the base location of the 32x8 tile number this thread is responsible of
    int tile_m = get_global_id(0) / 32 * MWG / MDIMC;
    int tile_n = get_global_id(1) * NWG / NDIMC;

    // the base pointers of agm, bgm and cgm
    const __global half * agm_ = agm + 32 * tile_m;
    const __global half * bgm_ = bgm + 8 * tile_n;
    __global half * cgm_ = cgm + kSizeM * 8 * tile_n + 32 * tile_m;

    // the (m,n) position within the warp
    int offset_number = laneid;
    int offset_m = offset_number % 8;
    int offset_n = offset_number / 8;
    
    if(laneid != get_global_id(0) % 32) {
        return;
    }

    int k, m, n, mb, nb, kb, kwg;
#ifdef USE_TC
    int zero_pair;
    asm("{\n"
        ".reg .b16 xh;\n"
        ".reg .b32 x;\n"
        "mov.f32 x, 0.0;\n"
        "cvt.rz.f16.f32 xh, x;\n"
        "mov.b32 %0, {xh,xh};\n"
        "}": "=r"(zero_pair)
    );

    int c0[MWG/MDIMC][NWG/NDIMC];
    int c1[MWG/MDIMC][NWG/NDIMC];
    int c2[MWG/MDIMC][NWG/NDIMC];
    int c3[MWG/MDIMC][NWG/NDIMC];
    for(mb = 0; mb < MWG / MDIMC; mb += 1) {
        for(nb = 0; nb < NWG / NDIMC; nb += 1) {
            c0[mb][nb] = zero_pair;
            c1[mb][nb] = zero_pair;
            c2[mb][nb] = zero_pair;
            c3[mb][nb] = zero_pair;
        }
    }
#else
    float acc[MWG/MDIMC][NWG/NDIMC][4][2];
    for(mb = 0; mb < MWG / MDIMC; mb += 1) {
        for(nb = 0; nb < NWG / NDIMC; nb += 1) {
            for(m=0; m<4; m++) {
                for(int n=0; n<2; n++) {
                    acc[mb][nb][m][n] = 0.0f;
                }
            }
        }
    }
#endif
    for(kwg = 0; kwg < kSizeK; kwg += 16 * KWG) {
#if SA == 1
    
#endif

#if SB == 1

#endif
#pragma unroll
        for(kb = 0; kb < 16 * KWG; kb += 16) {
#pragma unroll
            for(mb = 0; mb < MWG / MDIMC; mb += 1) {
#pragma unroll
                for(nb = 0; nb < NWG / NDIMC; nb += 1) {
                    const __global half * b_agm_ = agm_ + mb * 32;
                    const __global half * b_bgm_ = bgm_ + nb * 8;

                    const __global half * bb_agm_ = b_agm_ + kSizeM * (kb + kwg);
                    const __global half * bb_bgm_ = b_bgm_ + kSizeN * (kb + kwg);
#ifdef USE_TC
                    int d0 = c0[mb][nb];
                    int d1 = c1[mb][nb];
                    int d2 = c2[mb][nb];
                    int d3 = c3[mb][nb];
                    int c0_, c1_, c2_, c3_;
                    asm("{\n"
                        ".reg .b32 a0, a1, a2, a3, a4, a5, a6, a7;\n"
                        ".reg .b32 b0, b1, b2, b3, b4, b5, b6, b7;\n"
                        "wmma.load.a.sync.aligned.m32n8k16.row.f16 {a0,a1,a2,a3,a4,a5,a6,a7}, [%4], %6;\n"
                        "wmma.load.b.sync.aligned.m32n8k16.col.f16 {b0,b1,b2,b3,b4,b5,b6,b7}, [%5], %7;\n"
                        "wmma.mma.sync.aligned.row.col.m32n8k16.f16.f16 "
                        "    {%0,%1,%2,%3},\n"
                        "    {a0,a1,a2,a3,a4,a5,a6,a7},\n"
                        "    {b0,b1,b2,b3,b4,b5,b6,b7},\n"
                        "    {%8,%9,%10,%11};\n"
                        "}": "=r"(c0_), "=r"(c1_), "=r"(c2_), "=r"(c3_) : "l"(bb_agm_), "l"(bb_bgm_), "r"(kSizeM), "r"(kSizeN), "r"(d0), "r"(d1), "r"(d2), "r"(d3)
                    );
                    c0[mb][nb] = c0_;
                    c1[mb][nb] = c1_;
                    c2[mb][nb] = c2_;
                    c3[mb][nb] = c3_;
#else
                   for(m = offset_m; m < 32; m += 8) {
                       for(n = offset_n; n < 8; n += 4) {
                           float a = 0.0f;
                           for(k = 0; k < 16; k++) {
                               a += vload_half(kSizeM * k + m, bb_agm_) * vload_half(kSizeN * k + n, bb_bgm_);
                           }
                           acc[mb][nb][m/8][n/4] += a;
                       }
                   }
#endif
                }
            }
        }
    }

#ifdef USE_TC
    for(mb = 0; mb < MWG / MDIMC; mb += 1) {
        for(nb = 0; nb < NWG / NDIMC; nb += 1) {
            int c0_ = c0[mb][nb];
            int c1_ = c1[mb][nb];
            int c2_ = c2[mb][nb];
            int c3_ = c3[mb][nb];
            __global half * b_cgm_ = cgm_ + kSizeM * nb * 8 + mb * 32;
            asm("{\n"
                "wmma.store.d.sync.aligned.row.m32n8k16.f16 [%4], {%0,%1,%2,%3}, %5;"
                "}" : : "r"(c0_), "r"(c1_), "r"(c2_), "r"(c3_), "l"(b_cgm_), "r"(kSizeM));
        }
    }
#else
    for(mb = 0; mb < MWG / MDIMC; mb += 1) {
        for(nb = 0; nb < NWG / NDIMC; nb += 1) {
            for(m = offset_m; m < 32; m += 8) {
                for(n = offset_n; n < 8; n += 4) {
                    vstore_half(acc[mb][nb][m/8][n/4], kSizeM * (nb * 8 + n) + mb * 32 + m, cgm_);
                }
            }
        }
    }
#endif
}

__kernel void HgemmBatched(const int kSizeM, const int kSizeN, const int kSizeK,
                  const __global half* restrict agm,
                  const __global half* restrict bgm,
                  __global half* restrict cgm)
{
    // Sets the offsets
    const int batch = get_group_id(2);
    const int a_offset = kSizeM*kSizeK*batch;
    const int b_offset = kSizeK*kSizeN*batch;
    const int c_offset = kSizeM*kSizeN*batch;

    const __global half* restrict agm_ = &agm[a_offset];
    const __global half* restrict bgm_ = &bgm[b_offset];
    __global half* restrict cgm_ = &cgm[c_offset];

    // Allocates workgroup-private memory (local memory)
    #if SA == 1
      __local short alm[KWG * MWG * 256];
    #endif
    #if SB == 1
      __local short blm[KWG * NWG * 256];
    #endif

    #if SA == 1 && SB == 1
        HgemmBody(kSizeM, kSizeN, kSizeK, alm, blm, agm_, bgm_, cgm_);
    #elif SA == 1
        HgemmBody(kSizeM, kSizeN, kSizeK, alm, agm_, bgm_, cgm_);
    #elif SB == 1
        HgemmBody(kSizeM, kSizeN, kSizeK, blm, agm_, bgm_, cgm_);
    #else
        HgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_);
    #endif
}

