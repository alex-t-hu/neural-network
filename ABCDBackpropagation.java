import java.io.*;
import java.lang.String;

/*
 * Alex Hu
 * 11.29.21
 * Implements a multi-layer perceptron, without biases, using the Sigmoid activation function.
 * The perceptron can have any number of layers, all with a 
 * customizable number of activations. The perceptron can train itself with steepest gradient 
 * descent optimized with backpropagation to any set of inputs and outputs.
 * Uses files to determine the file output of the code.
 * 
 * quesiton 1: offset thing is this necessary, is it ok to have n-2 floating around somewhere
 * question 2: 
 */
public class ABCDBackpropagation
{
   static final double ONE_BILLION = 1000000000.0;
   static final int OFFSET_THETA = 1;                 //Theta's first element corresponds to the 2nd layer
   static final int OFFSET_PSI = 1;                   //Psi's first element corresponds to the 2nd layer
   
   int numLayers;                                     //technically unnecessary but stored for convenience
   double lambda;
   double lowerBoundForRandomWeights;
   double upperBoundForRandomWeights;
   boolean isRandomlyInitialized;
   
   int[] numActivations;                              //technically unnecessary but stored for convenience
   double[][] a;
   double[][] Theta;
   double[][] Psi;
   double[][][] w;
   
   /*
    * Constructor:
    * used when training and initial weights are randomly initialized
    * 
    * numLayers should equal numActivations.length
    * lowerBoundForRandomWeights must be less than upperBoundForRandomWeights
    * lambda should be greater than 0.0 and less than 1.0
    * 
    * isRandomlyInitialized should be true
    */
   public ABCDBackpropagation(int numLayers, int[] numActivations, double lowerBoundForRandomWeights, 
                              double upperBoundForRandomWeights, double lambda, boolean isRandomlyInitialized)
   {
      this.numLayers = numLayers;
      this.numActivations = numActivations;
      this.lambda = lambda;
      this.lowerBoundForRandomWeights = lowerBoundForRandomWeights;
      this.upperBoundForRandomWeights = upperBoundForRandomWeights;
      this.isRandomlyInitialized = isRandomlyInitialized;
      
      a = new double[numLayers][];
      Theta = new double[numLayers - OFFSET_THETA][];
      Psi = new double[numLayers - OFFSET_PSI][];
      w = new double[numLayers - 1][][];
      
      for (int alpha = 0; alpha < numLayers - 1; alpha++)            //is this ok?
      {
         a[alpha] = new double[numActivations[alpha]];
         Theta[alpha] = new double[numActivations[alpha + OFFSET_THETA]];
         Psi[alpha] = new double[numActivations[alpha + OFFSET_PSI]];
      }
      
      a[numLayers - 1] = new double[numActivations[numLayers - 1]];
      
      for (int n = 0; n < numLayers - 1; n++)                              //is this ok?
      {
         w[n] = new double[numActivations[n]][numActivations[n + 1]];
         
         for (int beta = 0; beta < numActivations[n]; beta++)
         {
            for (int gamma = 0; gamma < numActivations[n + 1]; gamma++)
            {
               w[n][beta][gamma] = Math.random() * (upperBoundForRandomWeights - lowerBoundForRandomWeights) 
                                   + lowerBoundForRandomWeights;
            }
         }
      } // for (int n = 0; n < numLayers - 1; n++)
   } // public ABCDBackpropagation(int numLayers, int[] numActivations, double lowerBoundForRandomWeights, ...

   /*
    * Constructor:
    * used when training and initial weights are pre-loaded 
    * 
    * the weights can be a jagged array
    * isRandomlyInitialized should be false
    */
   public ABCDBackpropagation(int numLayers, int[] numActivations, double[][][] w, 
                              double lambda, boolean isRandomlyInitialized)
   {
      this.numLayers = numLayers;
      this.numActivations = numActivations;
      this.lambda = lambda;
      this.isRandomlyInitialized = isRandomlyInitialized;
      this.w = w;
      
      a = new double[numLayers][];
      Theta = new double[numLayers - OFFSET_THETA][];
      Psi = new double[numLayers - OFFSET_PSI][];
      
      for (int alpha = 0; alpha < numLayers - 1; alpha++)
      {
         a[alpha] = new double[numActivations[alpha]];
         Theta[alpha] = new double[numActivations[alpha + OFFSET_THETA]];
         Psi[alpha] = new double[numActivations[alpha + OFFSET_PSI]];
      }
      
      a[numLayers - 1] = new double[numActivations[numLayers - 1]];
   } // public ABCDBackpropagation(int numLayers, int[] numActivations, double[][][] w, ...
   
   /*
    * Constructor:
    * used for just running based on pre-loaded weights
    * 
    * isRandomlyInitialized should be false
    */
   public ABCDBackpropagation(int numLayers, int[] numActivations, double[][][] w, 
                              boolean isRandomlyInitialized)
   {
      this.numLayers = numLayers;
      this.numActivations = numActivations;
      this.isRandomlyInitialized = isRandomlyInitialized;
      this.w = w;
      
      a = new double[numLayers][];
      
      for (int alpha = 0; alpha < numLayers; alpha++)
      {
         a[alpha] = new double[numActivations[alpha]];
      }
   } // public ABCDBackpropagation(int numLayers, int[] numActivations, double[][][] w, ...
   
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
    * runs the neural network quickly
    * calculates the activations for each layer to the 
    * right of the input layer
    * 
    * inputs is unchanged
    * inputs.length should be numActivations[0]
    */
   public void run(double[] inputs)
   {
      a[0] = inputs;
      
      for (int alpha = 1; alpha < numLayers; alpha++)          // is this ok?
      {
         for (int beta = 0; beta < numActivations[alpha]; beta++)
         {
            double theta = 0.0;
            
            for (int gamma = 0; gamma < numActivations[alpha - 1]; gamma++)
            {
               theta += a[alpha - 1][gamma] * w[alpha - 1][gamma][beta];
            }
            
            a[alpha][beta] = f(theta);
         } // for (int beta = 0; beta < numActivations[alpha]; beta++)
      } // for (int alpha = 1; alpha < numLayers; alpha++)
   } //public void run(double[] inputs)
   
   /*
    * optimizedTrain function:
    * steepest gradient descent optimized with backpropagation
    * trains to one set of inputs and outputs
    * 
    * inputs and T are not changed by this function
    */
   public void optimizedTrain(double[] inputs, double[] T)
   {
      a[0] = inputs;
      
      // runs the network, storing Theta values
      for (int alpha = 1; alpha < numLayers; alpha++)
      {
         for (int beta = 0; beta < numActivations[alpha]; beta++)
         {
            Theta[alpha - OFFSET_THETA][beta] = 0.0;
            
            for (int gamma = 0; gamma < numActivations[alpha - 1]; gamma++)
            {
               Theta[alpha - OFFSET_THETA][beta] += a[alpha - 1][gamma] * w[alpha - 1][gamma][beta];
            }
            
            a[alpha][beta] = f(Theta[alpha - OFFSET_THETA][beta]);
         } // for (int beta = 0; beta < numActivations[alpha]; beta++)
      } // for (int alpha = 1; alpha < numLayers; alpha++)
      
      // calculates the rightmost Psi values
      for (int beta = 0; beta < numActivations[numLayers - 1]; beta++)
      {
         Psi[numLayers - 1 - OFFSET_PSI][beta] = (T[beta] - a[numLayers - 1][beta]) * 
                                                 fprime(Theta[numLayers - 1 - OFFSET_THETA][beta]);
      }
      
      // calculates hidden layer Psi values and updates all but the leftmost weights
      for (int alpha = numLayers - 2; alpha > 0; alpha--)                                 //is the 2 here ok? overlal, are the comments her eok?
      {
         for (int beta = 0; beta < numActivations[alpha]; beta++)
         {
            double Omega = 0.0;
            
            for (int gamma = 0; gamma < numActivations[alpha + 1]; gamma++)
            {
               Omega += Psi[alpha + 1 - OFFSET_PSI][gamma] * w[alpha][beta][gamma];
            }
            
            Psi[alpha - OFFSET_PSI][beta] = Omega * fprime(Theta[alpha - OFFSET_THETA][beta]);
            
            for (int gamma = 0; gamma < numActivations[alpha + 1]; gamma++)
            {
               w[alpha][beta][gamma] += lambda * Psi[alpha + 1 - OFFSET_PSI][gamma] * a[alpha][beta];
            }
         } // for (int beta = 0; beta < numActivations[alpha]; beta++)
      } // for (int alpha = numLayers - 2; alpha > 0; alpha--)
      
      // updates the leftmost weights
      for (int beta = 0; beta < numActivations[0]; beta++)
      {
         for (int gamma = 0; gamma < numActivations[1]; gamma++)
         {
            w[0][beta][gamma] += lambda * Psi[1 - OFFSET_PSI][gamma] * a[0][beta];
         }
      }
   } //public void optimizedTrain(double[] inputs, double[] T)
   
   /*
    * getError function:
    * returns the error between the true values and the output activations
    */
   public double getError(double[] T)
   {
      double Error = 0.0;
      
      for (int beta = 0; beta < numActivations[numLayers - 1]; beta++)
      {
         Error += 0.5 * (T[beta] - a[numLayers - 1][beta]) * (T[beta] - a[numLayers - 1][beta]);
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
      double E = errorThreshold + 1.0;
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
            
            weightsWriter.newLine();
            weightsWriter.write("Current error = " + E);
            weightsWriter.close();
         } // if (saveWeightInterval > 0 && n % saveWeightInterval == 0)
      } // while (n < Nmax && E > errorThreshold)
      
      long endTime = System.nanoTime();
      long duration = endTime - startTime;
      double durationInSeconds = (double) duration / ONE_BILLION;
      
      BufferedWriter fileWriter = new BufferedWriter(new FileWriter(new File("src/" + fileName)));
      
      fileWriter.write("Training");
      fileWriter.newLine();
      fileWriter.write("Time took = " + String.format("%.9f", durationInSeconds) + " seconds");
      fileWriter.newLine();

      saveWeights(fileWriter);
      
      fileWriter.newLine();
      fileWriter.write("lambda = " + lambda);
      fileWriter.newLine();
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
         
         for (int beta = 0; beta < numActivations[0]; beta++)
         {
            fileWriter.write(inputs[t][beta] + " ");
         }
         
         fileWriter.newLine();
         fileWriter.write("Outputs");
         fileWriter.newLine();
         
         for (int beta = 0; beta < numActivations[numLayers - 1]; beta++)
         {
            fileWriter.write(a[numLayers - 1][beta] + " ");
         }
         
         fileWriter.newLine();
         fileWriter.write("True value (T)");
         fileWriter.newLine();
         
         for (int beta = 0; beta < numActivations[numLayers - 1]; beta++)
         {
            fileWriter.write(T[t][beta] + " ");
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
      fileWriter.write("Num layers = " + numLayers);
      fileWriter.newLine();
      
      for (int alpha = 0; alpha < numLayers; alpha++)
      {
         fileWriter.write("Num activations in layer " + alpha + " = " + numActivations[alpha]);
         fileWriter.newLine();
      }
      
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
      
      for (int n = 0; n < numLayers - 1; n++)
      {
         fileWriter.write("Weights for n = " + n);
         fileWriter.newLine();
         
         for (int beta = 0; beta < numActivations[n]; beta++)
         {
            for (int gamma = 0; gamma < numActivations[n + 1]; gamma++)
            {
               fileWriter.write(w[n][beta][gamma] + " ");
            }
            
            fileWriter.newLine();
         }
      } // for (int n = 0; n < numLayers - 1; n++)
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
    * readWeights function:
    * reads the weights from a file
    * returns the weights
    * 
    * Assumes the reader is currently on the line "Weights for n = 0" and ends the
    * reader on the line just after the last line
    */
   public static double[][][] readWeights(BufferedReader fileReader, int numLayers, int[] numActivations) throws Exception
   {
      double[][][] w = new double[numLayers - 1][][];
      
      for (int n = 0; n < numLayers - 1; n++)
      {
         fileReader.readLine();
         
         w[n] = new double[numActivations[n]][numActivations[n + 1]];
         
         for (int beta = 0; beta < numActivations[n]; beta++)
         {
            String[] splittedLine = fileReader.readLine().split(" ");
            
            for (int gamma = 0; gamma < numActivations[n + 1]; gamma++)
            {
               w[n][beta][gamma] = Double.parseDouble(splittedLine[gamma]);
            }
         }
      } // for (int n = 0; n < numLayers - 1; n++)
      
      return w;
   } // public static double[][][] readWeights(BufferedReader fileReader, int numLayers, ...
   
   /*
    * runOrTrainFromFile function:
    * runs or trains the neural network based on the control file
    */
   public static void runOrTrainFromFile(String inputFileName) throws Exception
   {
      ABCDBackpropagation perceptron;
      
      BufferedReader fileReader = new BufferedReader(new FileReader("src/" + inputFileName));
      boolean isTraining = fileReader.readLine().equals("Training");
      
      int numLayers = Integer.parseInt(extractLast(fileReader.readLine()));
      int[] numActivations = new int[numLayers];
      
      for (int alpha = 0; alpha < numLayers; alpha++)
      {
         numActivations[alpha] = Integer.parseInt(extractLast(fileReader.readLine()));
      }
      
      fileReader.readLine();
      
      if (isTraining)
      {
         double lambda = Double.parseDouble(extractLast(fileReader.readLine()));
         double errorThreshold = Double.parseDouble(extractLast(fileReader.readLine()));
         int Nmax = Integer.parseInt(extractLast(fileReader.readLine()));
         String outputFileName = extractLast(fileReader.readLine());
         int saveWeightsInterval = Integer.parseInt(extractLast(fileReader.readLine()));
         
         fileReader.readLine();
         
         boolean isRandomlyInitialized = Boolean.parseBoolean(extractLast(fileReader.readLine()));
         
         if (isRandomlyInitialized)
         {
            double lowerBoundForRandomWeights = Double.parseDouble(extractLast(fileReader.readLine()));
            double upperBoundForRandomWeights = Double.parseDouble(extractLast(fileReader.readLine()));
            perceptron = new ABCDBackpropagation(numLayers, numActivations, lowerBoundForRandomWeights, 
                                                 upperBoundForRandomWeights, lambda,
                                                 isRandomlyInitialized);
         }
         else
         {
            fileReader.readLine();

            perceptron = new ABCDBackpropagation(numLayers, numActivations, 
                                                readWeights(fileReader, numLayers, numActivations),
                                                lambda, isRandomlyInitialized);
         } // else
         
         fileReader.readLine();
         int numTestCases = Integer.parseInt(extractLast(fileReader.readLine()));
         double[][] inputs = new double[numTestCases][numActivations[0]];
         double[][] T = new double[numTestCases][numActivations[numLayers - 1]];
         
         for (int t = 0; t < numTestCases; t++)
         {
            String[] splitted = fileReader.readLine().split(" ");
            
            for (int beta = 0; beta < numActivations[0]; beta++)
            {
               inputs[t][beta] = Double.parseDouble(splitted[beta]);
            }
            
            splitted = fileReader.readLine().split(" ");
            
            for (int beta = 0; beta < numActivations[numLayers - 1]; beta++)
            {
               T[t][beta] = Double.parseDouble(splitted[beta]);
            }
            
            fileReader.readLine();
         } // for (int t = 0; t < numTestCases; t++)
         
         fileReader.close();
         
         perceptron.train(inputs, T, Nmax, errorThreshold, outputFileName,
                          saveWeightsInterval);
      } // if (isTraining)
      else
      {
         boolean isRandomlyInitialized = false;
         String outputFileName = extractLast(fileReader.readLine());
         
         fileReader.readLine();

         double[][][] w = readWeights(fileReader, numLayers, numActivations);
         double[] inputs = new double[numActivations[0]];
         
         fileReader.readLine();
         fileReader.readLine();
         
         String[] splittedLine = fileReader.readLine().split(" ");
         
         for (int beta = 0; beta < numActivations[0]; beta++)
         {
            inputs[beta] = Double.parseDouble(splittedLine[beta]);
         }
         
         fileReader.close();
         
         perceptron = new ABCDBackpropagation(numLayers, numActivations, w,
                                              isRandomlyInitialized);
         
         perceptron.runAndWriteToFile(inputs, outputFileName);
      } // else
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
      
      saveWeights(fileWriter);
      
      fileWriter.newLine();
      fileWriter.write("Input");
      fileWriter.newLine();
      
      for (int beta = 0; beta < numActivations[0]; beta++)
      {
         fileWriter.write(inputs[beta] + " ");
      }
      
      fileWriter.newLine();
      fileWriter.write("Outputs");
      fileWriter.newLine();
      
      for (int beta = 0; beta < numActivations[numLayers - 1]; beta++)
      {
         fileWriter.write(a[numLayers - 1][beta] + " ");
      }
      
      fileWriter.close();
   } // public void runAndWriteToFile(double[] inputs, String fileName) throws Exception
   
   /*
    * main function:
    * runs or trains the neural network based on the control file specified by 
    * the command line argument or a default file if no command line argument is passed
    */
   public static void main(String args[]) throws Exception
   {
      String defaultFileName = "trainXORWith2-2-3-3.txt";
      
      if (args.length == 0)
      {
         runOrTrainFromFile(defaultFileName);
      }
      else
      {
         runOrTrainFromFile(args[0]);
      }
   }
} // public class ABCDBackpropagation