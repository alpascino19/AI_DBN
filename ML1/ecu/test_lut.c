/* Verifică vigil_fuzzy_lut.h: citește "a risk conf" pe linii de la stdin, scrie cele 3 ieșiri.
 *   gcc -std=c99 -Wall -Wextra -O2 -o test_lut test_lut.c
 * Testul Python (ML1/tests/test_ecu.py) compară ieșirea cu ml1.ecu_fuzzy.lut_eval. */
#include <stdio.h>
#include "vigil_fuzzy_lut.h"

int main(void)
{
    float a, r, c, out[3];
    while (scanf("%f %f %f", &a, &r, &c) == 3) {
        vigil_fuzzy_eval(a, r, c, out);
        printf("%.5f %.5f %.5f\n", out[0], out[1], out[2]);
    }
    return 0;
}
