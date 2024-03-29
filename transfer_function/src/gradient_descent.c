#include <math.h>
#include <stdio.h>
// Precompile this script using the below in order to be called from Python
//cc -fPIC -shared -o cfuncs.so gradient_descent.c
void gradf(int N, double h1, double h2, double beta, double hE, const double *eta1, const double *etasim, double *grad, double *cost){
  double etap[N];
  double dot;
  double ratio;
  ratio = pow(h1/h2, 0.25);
  dot = 0.0;
  *cost = 0.0;
  for (int i=0;i<N; i++){
     etap[i] = ratio *(1.0f + beta*(hE-h2)/hE)*eta1[i];
     dot = dot + (eta1[i] * (etasim[i] - etap[i]));
     *cost = *cost + (((etasim[i] - etap[i]))*((etasim[i] - etap[i])));
  }
  *grad = (-2.0/(double)N)*ratio*((hE-h2)/hE)*dot;
  *cost = *cost/((double)N);
}
double momentum(int N, double h1, double h2, double beta, double hE, double learning_rate, double epsilon, const double *eta1, const double *etasim) {
	int n = 0;
	double norm, cost_old;
	gradf(N,h1,h2,beta,hE,eta1,etasim, &norm, &cost_old);
	double Vold, Vt, cost_new, diff;
	Vold = 0.0;
	cost_old = 2*cost_old;
	diff = 1.0;
	while((diff>epsilon) && n < 10000){
		gradf(N,h1,h2,beta,hE,eta1,etasim, &norm, &cost_new);
		Vt = 0.8*Vold + 0.2*norm;
		beta=beta-learning_rate*Vt;
		n=n+1;
		Vold = Vt;
		diff = cost_old - cost_new;
		cost_old = cost_new;
	}
	return beta;
}

void grad_alpha(int N, double h1, double h2, double alpha, const double *eta1, const double *etasim,double *grad, double *cost){
   double etap[N];
   double dot;
   double ratio;
   ratio = pow(h1/h2, 0.25);
   dot = 0.0;
   *cost = 0.0;
   for (int i=0;i<N; i++){
      etap[i] = ratio *(alpha)*eta1[i];
      // MSE
      dot = dot + (eta1[i] * (etasim[i] - etap[i]));
      *cost = *cost + ((etasim[i] - etap[i])*(etasim[i] - etap[i]));
      // RMSE
      // dot = dot + (eta1[i] * (etasim[i] - etap[i])/(sqrt((etasim[i] - etap[i])*(etasim[i] - etap[i])/((double)N))));
      // *cost = *cost + sqrt((etasim[i] - etap[i])*(etasim[i] - etap[i])/((double)N));
      // // MSRE
      // dot = dot + (eta1[i] * (etasim[i] - etap[i])/(etasim[i]));
      // *cost = *cost + ((etasim[i] - etap[i])*(etasim[i] - etap[i])/(etasim[i]*etasim[i]));

      // *cost = *cost + fabs(etasim[i] - etap[i])/(etasim[i]);
      // cost = cost + pow((etasim[i] - etap[i]),2.0);
   }
   //cost = (1/N)*cost;
   *grad = (-2.0/(double)N)*ratio*dot;
   // *grad = (-1.0/(double)N)*ratio*dot;
   // *cost = *cost;
   *cost = *cost/((double)N);
 }

 double lalli(int N, double h1, double h2, double alpha, double learning_rate, double epsilon, const double *eta1, const double *etasim) {
   int n = 0;
   double norm, cost_old;;//,cost_old,cost_new,diff;
   grad_alpha(N,h1,h2,alpha,eta1,etasim, &norm, &cost_old);
   double Vold, Vt, cost_new, diff;
   Vold = 0.0;
   cost_old = 2*cost_old;
   diff = 1.0;
   while((fabs(norm)>epsilon) && (n < 5000)){
     grad_alpha(N,h1,h2,alpha,eta1,etasim, &norm, &cost_new);
     alpha=alpha-learning_rate*norm;
     n=n+1;
     diff = fabs(cost_old - cost_new);
     cost_old = cost_new;
   }
   return alpha;
  }
// double fixedstep(int N, double h1, double h2, double beta, double hE, double learning_rate, double epsilon, const double *eta1, const double *etasim) {
//   int n = 0;
//   double norm = gradf(N,h1,h2,beta,hE,eta1,etasim);
//
//   while((fabs(norm)>epsilon) && n < 10000){
//     norm = gradf(N,h1,h2,beta,hE,eta1,etasim);
//     beta=beta-learning_rate*norm;
//     n=n+1;
// 	}
//   return beta;
//  }
  int main() {
   return 0;
}
