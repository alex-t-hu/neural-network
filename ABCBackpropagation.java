import java.io.*;
import java.lang.String;

/*
 * Alex Hu
 * 11.9.21
 * Implements a multi-layer perceptron, without biases, using Sigmoid activation functions.
 * The perceptron has an input layer, 1 hidden layer, and an output layer, all with a 
 * customizable number of activations. The perceptron can train itself with steepest gradient 
 * descent optimized with backpropagation to any set of inputs and outputs.
 * Uses files to determine the file output of the code.
 */
public class ABCBackpropagation
{
   static final double ONE_BILLION = 1000000000.0;
   
   int numInputActivations;
   int numHiddenLayerActivations;
   int numOutputActivations;
   double[] a;
   double[] Thetaj;
   double[] h;
   double[] Thetai;
   double[] F;
   double[][] w0;
   double[][] w1;
   double[] psi;
   double[] Omega;
   double lambda;
   double lowerBoundForRandomWeights;
   double upperBoundForRandomWeights;
   double E;
   long duration;
   boolean isRandomlyInitialized;
   
   /*
    * Constructor:
    * used when initial weights are randomly initialized
    * 
    * numInputActivations must be at least 1
    * numHiddenLayerActivations must be at least 1
    * numOutputActivations must be at least 1
    * lowerBoundForRandomWeights must be less than upperBoundForRandomWeights
    * lambda should be greater than 0.0 and less than 1.0
    * 
    * isRandomlyInitialized should be true
    */
   public ABCBackpropagation(int numInputActivations, int numHiddenLayerActivations,
                             int numOutputActivations, double lowerBoundForRandomWeights, 
                             double upperBoundForRandomWeights, double lambda, boolean isRandomlyInitialized)
   {
      this.numInputActivations = numInputActivations;
      this.numHiddenLayerActivations = numHiddenLayerActivations;
      this.numOutputActivations = numOutputActivations;
      this.lambda = lambda;
      this.lowerBoundForRandomWeights = lowerBoundForRandomWeights;
      this.upperBoundForRandomWeights = upperBoundForRandomWeights;
      this.isRandomlyInitialized = isRandomlyInitialized;
      
      a = new double[numInputActivations];
      h = new double[numHiddenLayerActivations];
      Thetaj = new double[numHiddenLayerActivations];
      Thetai = new double[numOutputActivations];
      w0 = new double[numInputActivations][numHiddenLayerActivations];
      w1 = new double[numHiddenLayerActivations][numOutputActivations];
      Omega = new double[numHiddenLayerActivations];
      psi = new double[numOutputActivations];
      F = new double[numOutputActivations];
      
      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenLayerActivations; j++)
         {
            w0[k][j] = Math.random() * (upperBoundForRandomWeights - lowerBoundForRandomWeights) + lowerBoundForRandomWeights;
         }
      }
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         for (int i = 0; i < numOutputActivations; i++)
         {
            w1[j][i] = Math.random() * (upperBoundForRandomWeights - lowerBoundForRandomWeights) + lowerBoundForRandomWeights;
         }    
      }
   } // public ABCPerceptron(int numInputActivations, int numHiddenLayerActivations, ...)

   /*
    * Constructor:
    * used when weights are pre-loaded
    * 
    * isRandomlyInitialized should be false
    */
   public ABCBackpropagation(int numInputActivations, int numHiddenLayerActivations,
                             int numOutputActivations, double lambda, double[][] w0, double[][] w1,
                             boolean isRandomlyInitialized)
   {
      this.numInputActivations = numInputActivations;
      this.numHiddenLayerActivations = numHiddenLayerActivations;
      this.numOutputActivations = numOutputActivations;
      this.lambda = lambda;
      this.isRandomlyInitialized = isRandomlyInitialized;
      
      a = new double[numInputActivations];
      h = new double[numHiddenLayerActivations];
      Thetaj = new double[numHiddenLayerActivations];
      Thetai = new double[numOutputActivations];
      this.w0 = w0;
      this.w1 = w1;
      Omega = new double[numHiddenLayerActivations];
      psi = new double[numOutputActivations];
      F = new double[numOutputActivations];
   } // public ABCPerceptron(int numInputActivations, int numHiddenLayerActivations, ...)
   
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
         Thetaj[j] = 0.0;
         
         for (int k = 0; k < numInputActivations; k++)
         {
            Thetaj[j] += w0[k][j] * a[k];
         }
         
         h[j] = f(Thetaj[j]);
      } // for (int j = 0; j < numHiddenLayerActivations; j++)
      
      for (int i = 0; i < numOutputActivations; i++)
      {
         Thetai[i] = 0.0;
         
         for (int j = 0; j < numHiddenLayerActivations; j++)
         {
            Thetai[i] += w1[j][i] * h[j];
         }
         
         F[i] = f(Thetai[i]);
      } // for (int i = 0; i < numOutputActivations; i++)
   } //public void run(double[] inputs)
   
   /*
    * optimizedTrain function:
    * steepest gradient descent optimized with backpropagation
    * trains to one set of inputs and outputs
    * 
    * inputs is unchanged
    * inputs.length should be numInputActivations
    * T is unchanged
    * T.length should be numOutputActivations
    */
   public void optimizedTrain(double[] inputs, double[] T)
   {
      a = inputs;
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         Thetaj[j] = 0.0;
         
         for (int k = 0; k < numInputActivations; k++)
         {
            Thetaj[j] += w0[k][j] * a[k];
         }
         
         h[j] = f(Thetaj[j]);
      } //for (int j = 0; j < numHiddenLayerActivations; j++)
      
      for (int i = 0; i < numOutputActivations; i++)
      {
         Thetai[i] = 0.0;
         
         for (int j = 0; j < numHiddenLayerActivations; j++)
         {
            Thetai[i] += w1[j][i] * h[j];
         }
         
         psi[i] = (T[i] - f(Thetai[i])) * fprime(Thetai[i]);
      } //for (int i = 0; i < numOutputActivations; i++)
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         Omega[j] = 0.0;
         
         for (int i = 0; i < numOutputActivations; i++)
         {
            Omega[j] += psi[i] * w1[j][i];
            w1[j][i] += lambda * h[j] * psi[i];
         }
      } //for (int j = 0; j < numHiddenLayerActivations; j++)
      
      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenLayerActivations; j++)
         {
            w0[k][j] += lambda * a[k] * Omega[j] * fprime(Thetaj[j]);
         }
      }
       
   } //public void optimizedTrain(double[] inputs, double[] T)
   
   /*
    * getError function:
    * returns the error between the true values, T, and the output, F
    */
   public double getError(double[] T)
   {
      double Error = 0.0;
      
      for (int i = 0; i < numOutputActivations; i++)
      {
         Error += 0.5 * (T[i] - F[i]) * (T[i] - F[i]);
      }
      
      return Error;
   } // public double getError(double[] T)
   
   /*
    * train function:
    * Repeatedly performs steepest gradient descent until Nmax iterations is reached 
    * or error is below errorThreshold
    * Prints the termination condition and error for each input
    * Saves the weights periodically or not if saveWeightInterval is less than 0
    * 
    * inputs is an array of inputs and T is the corresponding array of desired outputs
    * inputs and T are unchanged
    * 
    * inputs.length and T.length should be equal
    * errorThreshold should be at least 0.0
    * Nmax should be at least 1
    * numTestCases should be at least 1
    */
   public void train(double[][] inputs, double[][] T, int Nmax, double errorThreshold,
                     String fileName, int saveWeightInterval) throws Exception
   {
      int n = 0;
      int numTestCases = inputs.length;
      
      E = errorThreshold + 1;
      long startTime = System.nanoTime();

      
      while (n < Nmax && E > errorThreshold)
      {
         E = 0.0;
         
         for (int j = 0; j < numTestCases; j++)
         {
            optimizedTrain(inputs[j], T[j]);
         }
         
         n++;
         
         for (int j = 0; j < numTestCases; j++)
         {
            run(inputs[j]);
            
            E += getError(T[j]);
         }
         
         if (saveWeightInterval > 0 && n % saveWeightInterval == 0)
         {
            BufferedWriter weightsWriter = new BufferedWriter(new FileWriter(new File("src/Weights at n = " 
                                                                                      + n + " for " + fileName)));
            
            saveWeights(weightsWriter);
            
            weightsWriter.write("Current error = " + E);
            weightsWriter.close();
         } // if (saveWeightInterval > 0 && n % saveWeightInterval == 0)
      } // while (n < Nmax && E > errorThreshold)
      
      long endTime = System.nanoTime();
      
      duration = endTime - startTime;
      
      double durationInSeconds = (double) duration / ONE_BILLION;
      
      BufferedWriter fileWriter = new BufferedWriter(new FileWriter(new File("src/" + fileName)));
      
      fileWriter.write("Training");
      fileWriter.newLine();
      fileWriter.write("Time took = " + String.format("%.9f", durationInSeconds) + " seconds");
      fileWriter.newLine();

      saveWeights(fileWriter);
      
      fileWriter.newLine();
      
      if (E <= errorThreshold)
      {
         fileWriter.write("Total error, " + E + " , less than " + errorThreshold);
         fileWriter.newLine();
         fileWriter.write("Took " + n + " steps");
         fileWriter.newLine();
      }
      else
      {
         fileWriter.write("Total error, " + E + " , greater than " + errorThreshold);
         fileWriter.newLine();
         fileWriter.write("Maximum iterations, " + Nmax + " , reached");
         fileWriter.newLine();
      }
      
      for (int t = 0; t < numTestCases; t++)
      {
         run(inputs[t]);
         
         fileWriter.newLine();
         fileWriter.write("Input test case " + t);
         fileWriter.newLine();
         
         for (int k = 0; k < numInputActivations; k++)
         {
            fileWriter.write(inputs[t][k] + " ");
         }
         
         fileWriter.newLine();
         fileWriter.write("Outputs (F)");
         fileWriter.newLine();
         
         for (int i = 0; i < numOutputActivations; i++)
         {
            fileWriter.write(F[i] + " ");
         }
         
         fileWriter.newLine();
         fileWriter.write("True value (T)");
         fileWriter.newLine();
         
         for (int i = 0; i < numOutputActivations; i++)
         {
            fileWriter.write(T[t][i] + " ");
         }

         fileWriter.newLine();
      } // for (t = 0; t < numTestCases; t++)
      
      fileWriter.close();
   } // public void train(double[][] inputs, double[][] T, int Nmax, double errorThreshold, ...
   
   /*
    * saveWeights function:
    * Saves the network architecture and weights to a file
    * 
    * Does not close the fileWriter
    */
   public void saveWeights(BufferedWriter fileWriter) throws Exception
   {
      fileWriter.write("Num input activations = " + numInputActivations);
      fileWriter.newLine();
      fileWriter.write("Num hidden layer activations = " + numHiddenLayerActivations);
      fileWriter.newLine();
      fileWriter.write("Num output activations = " + numOutputActivations);
      
      fileWriter.newLine();
      fileWriter.newLine();
      
      fileWriter.write("isRandomlyInitialized = " + isRandomlyInitialized);
      fileWriter.newLine();
      
      if (isRandomlyInitialized)
      {
         fileWriter.write("Random weight range = ( " + lowerBoundForRandomWeights + " , " +
                          upperBoundForRandomWeights + " )");
         fileWriter.newLine();
      }

      fileWriter.newLine();
      fileWriter.write("Weights for n=0");
      fileWriter.newLine();
      
      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenLayerActivations; j++)
         {
            fileWriter.write(w0[k][j] + " ");
         }
         
         fileWriter.newLine();
      }
      
      fileWriter.write("Weights for n=1");
      fileWriter.newLine();
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         for (int i = 0; i < numOutputActivations; i++)
         {
            fileWriter.write(w1[j][i] + " ");
         }
         
         fileWriter.newLine();
      }
      
      fileWriter.newLine();
      fileWriter.write("lambda = " + lambda);
      fileWriter.newLine();
   } // public void saveWeights(BufferedWriter fileWriter) throws Exception
   
   /*
    * extractLast function:
    * In a string consisting of a sequence of words,
    * returns the last word in the string
    */
   public static String extractLast(String str)
   {
      String[] splitted = str.split(" ");
      
      return splitted[splitted.length - 1];
   }
   
   /*
    * runOrTrainFromFile function:
    * runs or trains the neural network based on the control file
    */
   public static void runOrTrainFromFile(String inputFileName) throws Exception
   {
      ABCBackpropagation perceptron;
      
      BufferedReader fileReader = new BufferedReader(new FileReader("src/" + inputFileName));
      
      if (fileReader.readLine().equals("Training"))
      {
         double lowerBoundForRandomWeights;
         double upperBoundForRandomWeights;
         double[][] w0;
         double[][] w1;
         
         int numInputActivations = Integer.parseInt(extractLast(fileReader.readLine()));
         int numHiddenLayerActivations = Integer.parseInt(extractLast(fileReader.readLine()));
         int numOutputActivations = Integer.parseInt(extractLast(fileReader.readLine()));
         
         fileReader.readLine();
         
         double lambda = Double.parseDouble(extractLast(fileReader.readLine()));
         double errorThreshold = Double.parseDouble(extractLast(fileReader.readLine()));
         int Nmax = Integer.parseInt(extractLast(fileReader.readLine()));
         String outputFileName = extractLast(fileReader.readLine());
         int saveWeightsInterval = Integer.parseInt(extractLast(fileReader.readLine()));
         
         fileReader.readLine();
         
         boolean isRandomlyInitialized = Boolean.parseBoolean(extractLast(fileReader.readLine()));
         
         if (isRandomlyInitialized)
         {
            lowerBoundForRandomWeights = Double.parseDouble(extractLast(fileReader.readLine()));
            upperBoundForRandomWeights = Double.parseDouble(extractLast(fileReader.readLine()));
            perceptron = new ABCBackpropagation(numInputActivations, numHiddenLayerActivations,
                                                numOutputActivations, lowerBoundForRandomWeights, 
                                                upperBoundForRandomWeights, lambda,
                                                isRandomlyInitialized);
         }
         else
         {
            w0 = new double[numInputActivations][numHiddenLayerActivations];
            w1 = new double[numHiddenLayerActivations][numOutputActivations];
            
            fileReader.readLine();
            fileReader.readLine();
            
            for (int k = 0; k < numInputActivations; k++)
            {
               String[] splittedLine = fileReader.readLine().split(" ");
               
               for (int j = 0; j < numHiddenLayerActivations; j++)
               {
                  w0[k][j] = Double.parseDouble(splittedLine[j]);
               }
            }
            
            fileReader.readLine();
            
            for (int j = 0; j < numHiddenLayerActivations; j++)
            {
               String[] splittedLine = fileReader.readLine().split(" ");
               
               for (int i = 0; i < numOutputActivations; i++)
               {
                  w1[j][i] = Double.parseDouble(splittedLine[i]);
               }
            }
            
            perceptron = new ABCBackpropagation(numInputActivations, numHiddenLayerActivations,
                                                numOutputActivations, lambda, w0, w1,
                                                isRandomlyInitialized);
         } // else
         
         fileReader.readLine();
         
         int numTestCases = Integer.parseInt(extractLast(fileReader.readLine()));
         double[][] inputs = new double[numTestCases][numInputActivations];
         double[][] T = new double[numTestCases][numOutputActivations];
         
         for (int n = 0; n < numTestCases; n++)
         {
            String[] splitted = fileReader.readLine().split(" ");
            
            for (int k = 0; k < numInputActivations; k++)
            {
               inputs[n][k] = Double.parseDouble(splitted[k]);
            }
            
            splitted = fileReader.readLine().split(" ");
            
            for (int i = 0; i < numOutputActivations; i++)
            {
               T[n][i] = Double.parseDouble(splitted[i]);
            }
            
            fileReader.readLine();
         } // for (int n = 0; n < numTestCases; n++)
         
         fileReader.close();
         
         perceptron.train(inputs, T, Nmax, errorThreshold, outputFileName,
                          saveWeightsInterval);
      } // if (fileReader.readLine().equals("Training"))
      else
      {
         double lambda = 0.0;
         boolean isRandomlyInitialized = false;
         
         int numInputActivations = Integer.parseInt(extractLast(fileReader.readLine()));
         int numHiddenLayerActivations = Integer.parseInt(extractLast(fileReader.readLine()));
         int numOutputActivations = Integer.parseInt(extractLast(fileReader.readLine()));
         
         fileReader.readLine();
         
         String outputFileName = extractLast(fileReader.readLine());
         
         double[][] w0 = new double[numInputActivations][numHiddenLayerActivations];
         double[][] w1 = new double[numHiddenLayerActivations][numOutputActivations];
         
         fileReader.readLine();
         fileReader.readLine();
         
         for (int k = 0; k < numInputActivations; k++)
         {
            String[] splittedLine = fileReader.readLine().split(" ");
            
            for (int j = 0; j < numHiddenLayerActivations; j++)
            {
               w0[k][j] = Double.parseDouble(splittedLine[j]);
            }
         }
         
         fileReader.readLine();
         
         for (int j = 0; j < numHiddenLayerActivations; j++)
         {
            String[] splittedLine = fileReader.readLine().split(" ");
            
            for (int i = 0; i < numOutputActivations; i++)
            {
               w1[j][i] = Double.parseDouble(splittedLine[i]);
            }
         }
         
         double[] inputs = new double[numInputActivations];
         
         fileReader.readLine();
         fileReader.readLine();
         
         String[] splittedLine = fileReader.readLine().split(" ");
         
         for (int k = 0; k < numInputActivations; k++)
         {
            inputs[k] = Double.parseDouble(splittedLine[k]);
         }
         
         fileReader.close();
         
         perceptron = new ABCBackpropagation(numInputActivations, numHiddenLayerActivations,
                                             numOutputActivations, lambda, w0, w1,
                                             isRandomlyInitialized);
         
         perceptron.runAndWriteToFile(inputs, outputFileName);
      } // else
      
      fileReader.close();
   } // public static void runOrTrainFromFile(String inputFileName) throws Exception
   
   /*
    * runAndWriteToFile function:
    * runs the inputs through the neural network and writes the outputs
    * and other information to the file
    * 
    * inputs should have length numInputActivations
    */
   public void runAndWriteToFile(double[] inputs, String fileName) throws Exception
   {
      run(inputs);
      
      BufferedWriter fileWriter = new BufferedWriter(new FileWriter(new File("src/" + fileName)));
      
      fileWriter.write("Num input activations = " + numInputActivations);
      fileWriter.newLine();
      fileWriter.write("Num hidden layer activations = " + numHiddenLayerActivations);
      fileWriter.newLine();
      fileWriter.write("Num output activations = " + numOutputActivations);
      fileWriter.newLine();
      fileWriter.newLine();
      
      fileWriter.write("Weights for n=0");
      fileWriter.newLine();
      
      for (int k = 0; k < numInputActivations; k++)
      {
         for (int j = 0; j < numHiddenLayerActivations; j++)
         {
            fileWriter.write(w0[k][j] + " ");
         }
         
         fileWriter.newLine();
      }
      
      fileWriter.write("Weights for n=1");
      fileWriter.newLine();
      
      for (int j = 0; j < numHiddenLayerActivations; j++)
      {
         for (int i = 0; i < numOutputActivations; i++)
         {
            fileWriter.write(w1[j][i] + " ");
         }
         
         fileWriter.newLine();
      }
      
      fileWriter.newLine();
      
      fileWriter.write("Input");
      fileWriter.newLine();
      
      for (int k = 0; k < numInputActivations; k++)
      {
         fileWriter.write(inputs[k] + " ");
      }
      
      fileWriter.newLine();
      fileWriter.write("Outputs (F)");
      fileWriter.newLine();
      
      for (int i = 0; i < numOutputActivations; i++)
      {
         fileWriter.write(F[i] + " ");
      }
      
      fileWriter.close();
   } // public void runAndWriteToFile(double[] inputs, String fileName) throws Exception
   
   /*
    * runMasterControlFile function:
    * runs the perceptron based on the file name provided in the master control file
    */
   public static void runMasterControlFile(String inputFileName) throws Exception
   {
      BufferedReader fileReader = new BufferedReader(new FileReader("src/" + inputFileName));
      
      runOrTrainFromFile(fileReader.readLine());
      
      fileReader.close();
   }
   
   /*
    * main function:
    * runs or trains the neural network based on the instructions of the control file
    */
   public static void main(String args[]) throws Exception
   {
      String inputFileName = "masterControlFile.txt";
      
      runMasterControlFile(inputFileName);
   }
} // public class ABCBackpropagation