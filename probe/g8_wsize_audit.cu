/* Byte-count audit: is GEMMul8's quantized A smaller than the fp64 W it would replace? */
#include <cstdio>
#include <gemmul8.hpp>
int main()
{
    struct Case { const char *tag; size_t mloc; size_t NB; } cs[] = {
        {"2-node champ (N=460800,P4)", 115200, 4096},
        {"1-node champ (N=319488,P2)", 159744, 4096},
        {"4-node champ (N=622592,P4)", 155648, 4096},
    };
    printf("%-28s %8s %6s | %10s %10s | %10s %8s\n",
           "case", "mloc", "NB", "fp64 W GB", "A_lo GB", "ratio", "verdict");
    for (auto &c : cs) {
        for (int mod : {11, 12}) {
            size_t wa = 0, wb = 0;
            gemmul8::workSize<false, gemmul8::Backend::INT8>(
                c.mloc, 3072, c.NB, mod, /*skipA=*/true, false, &wa, &wb);
            /* the transportable payload = the num_mat int8 planes (+ sftA int16/row),
             * i.e. everything workSizeA covers except the skip_scalA spare plane */
            double planes_gb = (double)c.mloc * c.NB * mod / 1e9;
            double sft_gb    = (double)c.mloc * 2 * mod / 1e9; /* negligible */
            double fp64_gb   = (double)c.mloc * c.NB * 8 / 1e9;
            double payload   = planes_gb + sft_gb;
            printf("%-28s %8zu %6zu | %10.2f %10.2f | %9.2fx %8s   (m=%d, workSizeA=%.2f GB)\n",
                   c.tag, c.mloc, c.NB, fp64_gb, payload, payload / fp64_gb,
                   payload < fp64_gb ? "SAVES" : "BIGGER", mod, wa / 1e9);
        }
    }
    printf("\nfp64 = 8 B/elem.  GEMMul8 INT8 backend: num_mat(NUM_MODULI) = NUM_MODULI\n"
           "  -> A_lo = NUM_MODULI int8 planes = NUM_MODULI B/elem.\n"
           "  break-even needs NUM_MODULI < 8; accuracy floor measured at 11.\n");
    return 0;
}
