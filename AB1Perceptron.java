/*
 * Alex Hu
 * 10.8.21
 * Implements a multi-layer perceptron, without biases, using Sigmoid activation functions.
 * The perceptron has an input layer and 1 hidden layer, both with a customizable number of activations.
 * The perceptron also has a single output activation.
 * The perceptron can train itself with steepest gradient descent to any set of inputs and outputs.
 */
public class AB1Perceptron
{
   int numInputActivations;
   int numHiddenLayerActivations;
   double[] a;
   double[] thetaj;
   double[] h;
   double theta0;
   double F0;
   double[][] w0;
   double[][] dEdw0;
   double[][] deltaw0;
   double[] w1;
   double[] dEdw1;
   double[] deltaw1;
   double omega0;
   double psi0;
   double lambda;
   double[] Omegaj;
   double[] Psij;
   double lowerBoundForRandomWeights;
   double upperBoundForRandomWeights;
   double E;
   
   /*
    * Constructor:
    * numInputActivations must be at least 1
    * numHiddenLayerActivations must be at least 1
    * lowerBoundForRandomWeights must be less than upperBoundForRandomWeights
    * lambda should be greater than 0.0 and less than 1.0
    */
   public AB1Perceptron(int numInputActivations, int numHiddenLayerActivations,
                        double lowerBoundForRandomWeights, double upperBoundForRandomWeights, 
                        double lambda)
   {
      this.numInputActivations = numInputActivations;
      this.numHiddenLayerActivations = numHiddenLayerActivations;
      this.lambda = lambda;
      this.lowerBoundForRandomWeights = lowerBoundForRandomWeights;
      this.upperBoundForRandomWeights = upperBoundForRandomWeights;
      
      a = new double[numInputActivations];
      h = new double[numHiddenLayerActivations];
      thetaj = new double[numHiddenLayerActivations];
      w0 = new double[numInputActivations][numHiddenLayerActivations];
      dEdw0 = new double[numInputActivations][numHiddenLayerActivations];
      deltaw0 = new double[numInputActivations][numHiddenLayerActivations];
      w1 = new double[numHiddenLayerActivations];
      deltaw1 = new double[numHiddenLayerActivations];
      dEdw1 = new double[numHiddenLayerActivations];
      Omegaj = new double[numHiddenLayerActivations];
      Psij = new double[numHiddenLayerActivations];
      
      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenLayerActivations; j++)
         {
            w0[k][j] = Math.random() * (upperBoundForRandomWeights - lowerBoundForRandomWeights) + lowerBoundForRandomWeights;
         }
      }
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         w1[j] = Math.random() * (upperBoundForRandomWeights - lowerBoundForRandomWeights) + lowerBoundForRandomWeights;
      }
   } // public AB1Perceptron(int numInputActivations, int numHiddenLayerActivations, ... )
   
   /*
    * f function:
    * returns f(x) where f is the Sigmoid function
    */
   public double f(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }
   
   /*
    * fprime function:
    * returns f'(x) or df/dx where f(x) is the Sigmoid function
    */
   public double fprime(double x)
   {
      double fx = f(x);
      
      return fx * (1.0 - fx);
   }
   
   /*
    * run function:
    * calculates the activations for a given input layer
    * inputs is unchanged
    * inputs.length should be numInputActivations
    */
   public void run(double[] inputs)
   {
      a = inputs;
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         thetaj[j] = 0.0;
         
         for (int k = 0; k < numInputActivations; k++)
         {
            thetaj[j] += w0[k][j] * a[k];
         }
         
         h[j] = f(thetaj[j]);
      }
      
      theta0 = 0.0;
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         theta0 += w1[j] * h[j];
      }
      
      F0 = f(theta0);
   } //public void run(double[] inputs)
   
   /*
    * getError function:
    * returns the error between the true value, T0, and the output value, F0
    */
   public double getError(double T0)
   {
      return 0.5 * (T0 - F0) * (T0 - F0);
   }
   
   /*
    * adjustWeights function:
    * updates the weights based on the gradient of the error with respect to the weights
    * should be called after run(inputs) is called
    * assumes that the 
    */
   public void adjustWeights(double T0)
   {
      omega0 = T0 - F0;
      psi0 = omega0 * fprime(theta0);
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         dEdw1[j] = -h[j] * psi0;
         deltaw1[j] = -lambda * dEdw1[j];
      }
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         Omegaj[j] = psi0 * w1[j];
         Psij[j] = Omegaj[j] * fprime(thetaj[j]);
      }
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         for (int k = 0; k < numInputActivations; k++)
         {
            dEdw0[k][j] = -a[k] * Psij[j];
            deltaw0[k][j] = -lambda * dEdw0[k][j];
         }
      }
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         w1[j] += deltaw1[j];
      }
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         for (int k = 0; k < numInputActivations; k++)
         {
            w0[k][j] += deltaw0[k][j];
         }
      }
   } //public void adjustWeights(double T0)
   
   /*
    * printWeights function:
    * prints all the weights of this perceptron in order
    */
   public void printWeights()
   {
      System.out.println("Printing weights");
      System.out.println("Weights for n=0");
      
      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenLayerActivations; j++)
         {
            System.out.print(w0[k][j]);
            System.out.print(" ");
         }
         
         System.out.println();
      }
      
      System.out.println("Weights for n=1");
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         System.out.print(w1[j]);
         System.out.print(" ");
      }
      
      System.out.println();
   } // public void printWeights()
   
   /*
    * train function:
    * Repeatedly performs steepest gradient descent until Nmax iterations is reached 
    * or error is below errorThreshold
    * Prints the termination condition and error for each input
    *  
    * inputs is an array of inputs and T is the corresponding array of desired outputs
    * inputs and T are unchanged
    * 
    * inputs.length and T.length should be equal
    * errorThreshold should be at least 0.0
    * Nmax should be at least 1
    * numTestCases should be at least 1
    */
   public void train(double[][] inputs, double[] T, int Nmax, double errorThreshold)
   {
      int i = 0;
      int numTestCases = inputs.length;
      
      E = errorThreshold + 1;
      
      while (i < Nmax && E > errorThreshold)
      {
         E = 0.0;
         
         for (int j = 0; j < numTestCases; j++)
         {
            run(inputs[j]);
            adjustWeights(T[j]);
         }
         
         i++;
         
         for (int j = 0; j < numTestCases; j++)
         {
            run(inputs[j]);
            
            E += getError(T[j]);
         }
      } // while (i < Nmax && E > errorThreshold)
      
      System.out.println("Training");
      System.out.println();
      System.out.println("Num input activations = " + numInputActivations);
      System.out.println("Num hidden layer activations = " + numHiddenLayerActivations);
      System.out.println("Random weight range = (" + lowerBoundForRandomWeights + ", " +
                         upperBoundForRandomWeights + ")");
      
      printWeights();
      
      System.out.println("lambda = " + lambda);
      
      if (E <= errorThreshold)
      {
         System.out.println("Total error, " + E + " , less than " + errorThreshold);
         System.out.println("Took " + i + " steps");
      }
      else
      {
         System.out.println("Total error, " + E + " , greater than " + errorThreshold);
         System.out.println("Maximum iterations, " + Nmax + " , reached");
      }
      
      System.out.println();
      
      for (i = 0; i < numTestCases; i++)
      {
         run(inputs[i]);
         
         System.out.println("Input test case " + i);
         
         for (int k = 0; k < numInputActivations; k++)
         {
            System.out.print(inputs[i][k]);
            System.out.print(" ");
         }
         
         System.out.println();
         System.out.println("Output (F0) =  " + F0);
         System.out.println("True value (T0) =  " + T[i]);
         System.out.println();
      } // for (i = 0; i < numTestCases; i++)
   } // public void train(double[][] inputs, double[] T, int Nmax, double errorThreshold)
   
   /*
    * setWeights function:
    * sets the weights to w0 and w1
    * 
    * w0's dimensions should be numInputActivations by numHiddenLayerActivations
    * w1 should have length numHiddenLayerActivations
    */
   public void setWeights(double[][] w0, double[] w1)
   {
      this.w0 = w0;
      this.w1 = w1;
   }
   
   /*
    * runAndPrint function:
    * runs the inputs.
    * prints the output and information about the neural network
    * 
    * inputs should have length numInputActivations
    */
   public void runAndPrint(double[] inputs)
   {
      run(inputs);
      
      System.out.println("Running");
      System.out.println();
      System.out.println("Num input activations = " + numInputActivations);
      System.out.println("Num hidden layer activations = " + numHiddenLayerActivations);
      
      printWeights();
      
      System.out.println();
      
      System.out.println("Input");
      
      for (int k = 0; k < numInputActivations; k++)
      {
         System.out.print(inputs[k]);
         System.out.print(" ");
      }
      
      System.out.println();
      System.out.println("Output (F0) =  " + F0);
   } // public void runAndPrint(double[] inputs)
   
   /*
    * main:
    * creates the perceptron
    * can either run the perceptron on certain input data
    * or train the perceptron on multiple test cases
    * 
    * user can change the variable, isTraining, to determine whether
    * code trains or runs
    */
   public static void main(String args[])
   {
      boolean isTraining = false;
      
      int numInputActivations = 2;
      int numHiddenLayerActivations = 2;
      double lowerBoundForRandomWeights = -1.0;
      double upperBoundForRandomWeights = 1.5;
      double lambda = 0.1;
      
      double errorThreshold = 0.001;
      int Nmax = 1000000;
      
      double[][] w0 = {
                         {1.0, 2.0},
                         {-0.5, 1.5}
                      };
      double[] w1 = {1.5, -1.5};
      double[] singleInput = {0.0, 1.0};
      
      double[][] inputs = {
                             {0.0, 0.0},
                             {0.0, 1.0},
                             {1.0, 0.0},
                             {1.0, 1.0}
                          };
      double[] T = {0.0, 1.0, 1.0, 0.0};
      
      AB1Perceptron perceptron = new AB1Perceptron(numInputActivations, numHiddenLayerActivations,
                                                   lowerBoundForRandomWeights, upperBoundForRandomWeights, 
                                                   lambda);
      
      if (isTraining)
      {
         perceptron.train(inputs, T, Nmax, errorThreshold);
      }
      else
      {
         perceptron.setWeights(w0, w1);
         perceptron.runAndPrint(singleInput);
      }
   } // public static void main(String args[])
} // public class AB1Perceptron
