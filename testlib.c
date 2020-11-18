/* testlib.c */

#include <math.h>

double f(int n, double *x) {
  return 1/sqrt(x[1]*pow(1+x[0], 3) + x[3]*pow(1+x[0], 2) + x[2]*pow(1+x[0], 3*(1+x[4])));
}
