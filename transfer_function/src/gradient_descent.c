#include <math.h>
#include <stdio.h>
//cc -fPIC -shared -o cfuncs.so gradient_descent.c
void grad_alpha(int N, double h1, double h2, double alpha, const double *eta1, const double *etasim,double *grad, double *cost){
   double etap[N];
   double dot;
   double ratio;
   ratio = pow(h1/h2, 0.25);
   dot = 0.0;
   *cost = 0.0;
   for (int i=0;i<N; i++){
      etap[i] = ratio *(alpha)*eta1[i];
      dot = dot + (eta1[i] * (etasim[i] - etap[i]));
      *cost = *cost + (((etasim[i] - etap[i]))*((etasim[i] - etap[i])));
   }
   *grad = (-2.0/(double)N)*ratio*dot;
   *cost = *cost/((double)N);
 }

 double lalli(int N, double h1, double h2, double alpha, double learning_rate, double epsilon, const double *eta1, const double *etasim) {
   int n = 0;
   double norm, cost_old;
   grad_alpha(N,h1,h2,alpha,eta1,etasim, &norm, &cost_old);
   double Vold, Vt, cost_new, diff;
   Vold = 0.0;
   cost_old = 2*cost_old;
   diff = 1.0;
   while((diff>epsilon) && n < 10000){
     grad_alpha(N,h1,h2,alpha,eta1,etasim, &norm, &cost_new);
     Vt = 0.8*Vold + 0.2*norm;
     alpha=alpha-learning_rate*Vt;
     n=n+1;
     Vold = Vt;
     diff = cost_old - cost_new;
     cost_old = cost_new;
 	}
   return alpha;
  }
  int main() {
   return 0;
}
