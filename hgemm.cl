// #define USE_TC

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
#if 1
    int laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));

    // the base location of the 16x16 tile number this thread is responsible of
    int tile_m = get_global_id(0) / 32 * MWG / MDIMC;
    int tile_n = get_global_id(1) * NWG / NDIMC;

    // the base pointers of agm, bgm and cgm
    const __global half * agm_ = agm + 16 * tile_m;
    const __global half * bgm_ = bgm + 16 * tile_n;
    __global half * cgm_ = cgm + kSizeM * 16 * tile_n + 16 * tile_m;

    // the (m,n) position within the warp
    int offset_number = laneid;
    int offset_m = offset_number % 8;
    int offset_n = offset_number / 8;
    
    if(laneid != get_global_id(0) % 32) {
        return;
    }
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
#endif

    int k, m, n, mb, nb, kb, kwg;
#ifdef USE_TC
    int c0[MWG/MDIMC][NWG/NDIMC];
    int c1[MWG/MDIMC][NWG/NDIMC];
    int c2[MWG/MDIMC][NWG/NDIMC];
    int c3[MWG/MDIMC][NWG/NDIMC];
    for(mb = 0; mb < MWG / MDIMC; mb += 1) {
        for(nb = 0; nb < NWG / NDIMC; nb += 1) {
            c0[mb][nb] = zero_pair;
        }
    }
#else
    float acc[MWG/MDIMC][NWG/NDIMC][2][4];
    for(mb = 0; mb < MWG / MDIMC; mb += 1) {
        for(nb = 0; nb < NWG / NDIMC; nb += 1) {
            for(m=0; m<2; m++) {
                for(int n=0; n<4; n++) {
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
        for(kb = 0; kb < 16 * KWG; kb += 16) {
            for(mb = 0; mb < MWG / MDIMC * 16; mb += 16) {
                for(nb = 0; nb < NWG / NDIMC * 16; nb += 16) {
                    const __global half * b_agm_ = agm_ + mb;
                    const __global half * b_bgm_ = bgm_ + nb;

                    const __global half * bb_agm_ = b_agm_ + kSizeM * (kb + kwg);
                    const __global half * bb_bgm_ = b_bgm_ + kSizeN * (kb + kwg);
#ifdef USE_TC
                    asm("{\n"
                        ".reg .b32 a0, a1, a2, a3, a4, a5, a6, a7;\n"
                        ".reg .b32 b0, b1, b2, b3, b4, b5, b6, b7;\n"
                        ".reg .b32 c0, c1, c2, c3;\n"
                        "wmma.load.a.sync.aligned.m16n16k16.col.f16 {a0,a1,a2,a3,a4,a5,a6,a7}, [%4], %6;\n"
                        "wmma.load.b.sync.aligned.m16n16k16.row.f16 {b0,b1,b2,b3,b4,b5,b6,b7}, [%5], %7;\n"
                        "wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 "
                        "    {%0,%1,%2,%3},\n"
                        "    {a0,a1,a2,a3,a4,a5,a6,a7},\n"
                        "    {b0,b1,b2,b3,b4,b5,b6,b7},\n"
                        "    {%0,%1,%2,%3};\n"
                        "}": "+r"(c0[mb/16][nb/16]), "+r"(c1[mb/16][nb/16]), "+r"(c2[mb/16][nb/16]), "+r"(c3[mb/16][nb/16]) : "l"(bb_agm_), "l"(bb_bgm_), "r"(kSizeM), "r"(kSizeN): "memory");
#else
                   for(m = offset_m; m < 16; m += 8) {
                       for(n = offset_n; n < 16; n += 4) {
                           float a = 0.0f;
                           for(k = 0; k < 16; k++) {
                               a += vload_half(kSizeM * k + m, bb_agm_) * vload_half(kSizeN * k + n, bb_bgm_);
                           }
                           acc[mb/16][nb/16][m/8][n/4] += a;
                       }
                   }
#endif
                }
            }
        }
    }

#ifdef USE_TC
    asm("{\n"
        "wmma.store.d.sync.aligned.col.m16n16k16.f16 [%4], {%0,%1,%2,%3}, %5;"
    "}" : : "r"(c0), "r"(c1), "r"(c2), "r"(c3), "l"(cgm_), "r"(kSizeM));
#else
    for(mb = 0; mb < MWG / MDIMC; mb += 1) {
        for(nb = 0; nb < NWG / NDIMC; nb += 1) {
            for(m = offset_m; m < 16; m += 8) {
                for(n = offset_n; n < 16; n += 4) {
                    vstore_half(acc[mb][nb][m/8][n/4], kSizeM * (nb * 16 + n) + mb * 16 + m, cgm_);
                }
            }
        }
    }
#endif

#else
    {
        int k, m, n;
        for(m=0; m<kSizeM; m++) {
            for(n=0; n<kSizeN; n++) {
                float acc = 0.0f;
                for(k=0; k < kSizeK; k++) {
                    acc += vload_half(kSizeM * k + m, agm) * vload_half(kSizeN * k + n, bgm);
                }
                vstore_half(acc, kSizeM * n + m, cgm);
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

