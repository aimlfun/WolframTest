using System.Security.Cryptography;
using System.Diagnostics;
using System.Text;
using System;

namespace WolframTest;

/// <summary>
///    _   _                      _   _   _      _                      _    
///   | \ | | ___ _   _ _ __ __ _| | | \ | | ___| |___      _____  _ __| | __
///   |  \| |/ _ \ | | | '__/ _` | | |  \| |/ _ \ __\ \ /\ / / _ \| '__| |/ /
///   | |\  |  __/ |_| | | | (_| | | | |\  |  __/ |_ \ V  V / (_) | |  |   < 
///   |_| \_|\___|\__,_|_|  \__,_|_| |_| \_|\___|\__| \_/\_/ \___/|_|  |_|\_\
///                                                                          
/// Implementation of a feedforward neural network.
///         
///   A neuron is simply:
///      output = SUM( weight * input ) + bias
///                "weight" amplifies or reduces the input it receives from a neuron that feeds into it. It is from the conceptual dendrite.
///                "bias" is how much is added to the neuron output. (think fires when it reaches a threshold, this lowers the need for the
///                neuron to fire for the output to be "on" full.
/// </summary>
public class NeuralNetwork
{
    /// </summary>
    /// Tracks the neural networks.
    /// <summary>
    internal static Dictionary<int, NeuralNetwork> s_networks = new();

    /// <summary>
    /// The "id" (index) of the brain, should also align to the "id" of the item it is attached.
    /// </summary>
    internal int Id;

    /// <summary>
    /// How many layers of neurons (3+). Do not do 1.
    /// 2 => input connected to output.
    /// 1 => input is output, and feed forward will crash.
    /// </summary>
    internal readonly int[] Layers;

    /// <summary>
    /// The neurons.
    /// [layer][neuron]
    /// </summary>
    internal double[][] Neurons;

    /// <summary>
    /// NN Biases. Either improves or lowers the chance of this neuron fully firing.
    /// [layer][neuron]
    /// </summary>
    internal double[][] Biases;

    /// <summary>
    /// NN weights. Reduces or amplifies the output for the relationship between neurons in each layer
    /// [layer][neuron][neuron]
    /// </summary>
    internal double[][][] Weights;

    /// <summary>
    /// Controls the speed of back-propagation (too large: oscillation will occur, too small: takes forever to train).
    /// </summary>
    private readonly float learningRate = 0.05f;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="_id">Unique ID of the neuron.</param>
    /// <param name="layerDefinition">Defines size of the layers.</param>
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
    internal NeuralNetwork(int _id, int[] layerDefinition)
#pragma warning restore CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
    {
        // (1) INPUT (2) HIDDEN (3) OUTPUT.
        if (layerDefinition.Length < 2) throw new ArgumentException(nameof(layerDefinition) + " insufficient layers.");

        Id = _id; // used to reference this network

        Layers = new int[layerDefinition.Length];

        for (int layer = 0; layer < layerDefinition.Length; layer++)
        {
            Layers[layer] = layerDefinition[layer];
        }

        // if layerDefinition is [2,3,2] then...
        // 
        // Neurons :      (o) (o)    <-2  INPUT
        //              (o) (o) (o)  <-3
        //                (o) (o)    <-2  OUTPUT
        //

        InitialiseNeurons();
        InitialiseBiases();
        InitialiseWeights();

        // track all the networks we created
        if (!s_networks.ContainsKey(Id)) s_networks.Add(Id, this); else s_networks[Id] = this;
    }

    /// <summary>
    /// Derivative (for back-propagation of TanH activation function).
    /// </summary>
    /// <param name="value"></param>
    /// <returns></returns>
    public static double DerivativeOfTanHActivationFunction(double value)
    {
        return 1 - value * value;
    }

    /// <summary>
    /// Create empty storage array for the neurons in the network.
    /// </summary>
    private void InitialiseNeurons()
    {
        List<double[]> neuronsList = new();

        for (int layer = 0; layer < Layers.Length; layer++)
        {
            neuronsList.Add(new double[Layers[layer]]);
        }

        Neurons = neuronsList.ToArray();
    }

    /// <summary>
    /// Generate a cryptographic random number between -0.5...+0.5.
    /// </summary>
    /// <returns></returns>
    private static float RandomFloatBetweenMinusHalfToPlusHalf()
    {
        return (float)(RandomNumberGenerator.GetInt32(0, 100000) - 50000) / 100000;
    }

    /// <summary>
    /// initializes and populates biases.
    /// </summary>
    private void InitialiseBiases()
    {
        List<double[]> biasList = new();

        // for each layer of neurons, we have to set biases.
        for (int layer = 1; layer < Layers.Length; layer++)
        {
            double[] bias = new double[Layers[layer]];

            for (int biasLayer = 0; biasLayer < Layers[layer]; biasLayer++)
            {
                bias[biasLayer] = RandomFloatBetweenMinusHalfToPlusHalf();
            }

            biasList.Add(bias);
        }

        Biases = biasList.ToArray();
    }

    /// <summary>
    /// initializes random array for the weights being held in the network.
    /// </summary>
    private void InitialiseWeights()
    {
        List<double[][]> weightsList = new(); // used to construct weights, as dynamic arrays aren't supported

        for (int layer = 1; layer < Layers.Length; layer++)
        {
            List<double[]> layerWeightsList = new();

            int neuronsInPreviousLayer = Layers[layer - 1];

            for (int neuronIndexInLayer = 0; neuronIndexInLayer < Neurons[layer].Length; neuronIndexInLayer++)
            {
                double[] neuronWeights = new double[neuronsInPreviousLayer];

                for (int neuronIndexInPreviousLayer = 0; neuronIndexInPreviousLayer < neuronsInPreviousLayer; neuronIndexInPreviousLayer++)
                {
                    neuronWeights[neuronIndexInPreviousLayer] = RandomFloatBetweenMinusHalfToPlusHalf();
                }

                layerWeightsList.Add(neuronWeights);
            }

            weightsList.Add(layerWeightsList.ToArray());
        }

        Weights = weightsList.ToArray();
    }

    /// <summary>
    /// Feed forward, inputs >==> outputs.
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    internal double[] FeedForward(double[] inputs)
    {
        // put the INPUT values into layer 0 neurons
        for (int i = 0; i < inputs.Length; i++)
        {
            Neurons[0][i] = inputs[i];
        }

        for (int layer = 1; layer < Layers.Length; layer++)
        {
            for (int neuronIndexForLayer = 0; neuronIndexForLayer < Layers[layer]; neuronIndexForLayer++)
            {
                double value = 0f;

                for (int neuronIndexInPreviousLayer = 0; neuronIndexInPreviousLayer < Layers[layer - 1]; neuronIndexInPreviousLayer++)
                {
                    value += Weights[layer - 1][neuronIndexForLayer][neuronIndexInPreviousLayer] * Neurons[layer - 1][neuronIndexInPreviousLayer];
                }

                Neurons[layer][neuronIndexForLayer] = Math.Tanh(value + Biases[layer - 1][neuronIndexForLayer]);
            }
        }

        return Neurons[^1]; // final* layer contains OUTPUT
    }

    /// <summary>
    /// Performs back propagation to train network.
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="expected"></param>
    public void BackPropagate(double[] inputs, double[] expected)
    {
        double[] output = FeedForward(inputs);// runs feed forward to ensure neurons are populated correctly

        List<double[]> gammaList = new();

        for (int i = 0; i < Layers.Length; i++)
        {
            gammaList.Add(new double[Layers[i]]);
        }

        double[][] gamma = gammaList.ToArray(); // gamma initialization

        int lastHIDDENLayer = Layers.Length - 2;
        int OUTPUTLayer = Layers.Length - 1;

        for (int neuronIndexOUTPUTLayer = 0; neuronIndexOUTPUTLayer < output.Length; neuronIndexOUTPUTLayer++)
        {
            gamma[OUTPUTLayer][neuronIndexOUTPUTLayer] = (output[neuronIndexOUTPUTLayer] - expected[neuronIndexOUTPUTLayer]) * DerivativeOfTanHActivationFunction(output[neuronIndexOUTPUTLayer]);

            Biases[lastHIDDENLayer][neuronIndexOUTPUTLayer] -= gamma[OUTPUTLayer][neuronIndexOUTPUTLayer] * learningRate;

            for (int neuronIndexInLastHIDDENLayer = 0; neuronIndexInLastHIDDENLayer < Layers[lastHIDDENLayer]; neuronIndexInLastHIDDENLayer++)
            {
                Weights[lastHIDDENLayer][neuronIndexOUTPUTLayer][neuronIndexInLastHIDDENLayer] -= gamma[OUTPUTLayer][neuronIndexOUTPUTLayer] * Neurons[lastHIDDENLayer][neuronIndexInLastHIDDENLayer] * learningRate;
            }
        }

        int previousHIDDENLayer = lastHIDDENLayer - 1;
        int nextHIDDENLayer = OUTPUTLayer;

        for (int i = lastHIDDENLayer; i > 0; i--) // runs on all hidden layers
        {
            for (int j = 0; j < Layers[i]; j++) // outputs
            {
                for (int k = 0; k < gamma[nextHIDDENLayer].Length; k++)
                {
                    gamma[i][j] += gamma[nextHIDDENLayer][k] * Weights[i][k][j];
                }

                gamma[i][j] *= DerivativeOfTanHActivationFunction(Neurons[i][j]); //calculate gamma

                Biases[previousHIDDENLayer][j] -= gamma[i][j] * learningRate; // modify biases of network

                for (int k = 0; k < Layers[previousHIDDENLayer]; k++) // iterate over inputs to layer
                {
                    Weights[previousHIDDENLayer][j][k] -= gamma[i][j] * Neurons[previousHIDDENLayer][k] * learningRate; // modify weights of network
                }
            }

            --previousHIDDENLayer;
            --nextHIDDENLayer;
        }
    }

    /// <summary>
    /// This loads the biases and weights from within a file into the neural network.
    /// </summary>
    /// <param name="path"></param>
    internal void Load(string path)
    {
        if (!File.Exists(path)) return;

        string[] ListLines = File.ReadAllLines(path);

        int index = 0;

        try
        {
            for (int layerIndex = 0; layerIndex < Biases.Length; layerIndex++)
            {
                for (int neuronIndex = 0; neuronIndex < Biases[layerIndex].Length; neuronIndex++)
                {
                    Biases[layerIndex][neuronIndex] = double.Parse(ListLines[index++]);
                }
            }

            for (int layerIndex = 0; layerIndex < Weights.Length; layerIndex++)
            {
                for (int neuronIndexInLayer = 0; neuronIndexInLayer < Weights[layerIndex].Length; neuronIndexInLayer++)
                {
                    for (int neuronIndexInPreviousLayer = 0; neuronIndexInPreviousLayer < Weights[layerIndex][neuronIndexInLayer].Length; neuronIndexInPreviousLayer++)
                    {
                        Weights[layerIndex][neuronIndexInLayer][neuronIndexInPreviousLayer] = double.Parse(ListLines[index++]);
                    }
                }
            }
        }
        catch (Exception)
        {
            MessageBox.Show("Unable to load .AI files\nThe most likely reason is that the number of neurons does not match the saved AI file.");
        }
    }

    /// <summary>
    /// Saves the trained model.
    /// </summary>
    internal static void SaveTrainedModel()
    {
        // each file is saved as a .AI file.
        foreach (int id in s_networks.Keys)
        {
            NeuralNetwork nn = s_networks[id];
            nn.Save(Path.Combine(@"c:\temp\", $"Wolfram{nn.Id}.ai"));
        }

        MessageBox.Show("AI model saved");
    }

    /// <summary>
    /// Loads the saved model.
    /// </summary>
    internal static void LoadTrainedModel()
    {
        // each file is saved as a .AI file.
        foreach (int id in s_networks.Keys)
        {
            NeuralNetwork nn = s_networks[id];
            nn.Load(Path.Combine(@"c:\temp\", $"Wolfram{nn.Id}.ai"));
        }

        MessageBox.Show("AI model loaded");
    }

    /// <summary>
    /// Saves the biases and weights within the network to a file.
    /// </summary>
    /// <param name="path"></param>
    internal void Save(string path)
    {
        using StreamWriter writer = new(path, false);

        // write the biases
        for (int layerIndex = 0; layerIndex < Biases.Length; layerIndex++)
        {
            for (int neuronIndex = 0; neuronIndex < Biases[layerIndex].Length; neuronIndex++)
            {
                writer.WriteLine(Biases[layerIndex][neuronIndex]);
            }
        }

        // write the weights
        for (int layerIndex = 0; layerIndex < Weights.Length; layerIndex++)
        {
            for (int neuronIndexInLayer = 0; neuronIndexInLayer < Weights[layerIndex].Length; neuronIndexInLayer++)
            {
                for (int neuronIndexInPreviousLayer = 0; neuronIndexInPreviousLayer < Weights[layerIndex][neuronIndexInLayer].Length; neuronIndexInPreviousLayer++)
                {
                    writer.WriteLine(Weights[layerIndex][neuronIndexInLayer][neuronIndexInPreviousLayer]);
                }
            }
        }

        writer.Close();
    }


    /// <summary>
    /// Returns the formula for the neural network.
    /// </summary>
    /// <returns></returns>
    internal string Formula()
    {
        if (Layers[0] != 1 || Layers[^1] != 1) return "Formula created for 1 input, 1 output";
        
        // single input of "x"
        Dictionary<string, string> values = new()
        {
            { $"0-0", $"x" }
        };

        for (int layer = 1; layer < Layers.Length; layer++)
        {
            List<string> dictionaryEntriesWeShouldRemove = new();

            for (int neuronIndexForLayer = 0; neuronIndexForLayer < Layers[layer]; neuronIndexForLayer++)
            {
                StringBuilder valueFormula = new(20);

                for (int neuronIndexInPreviousLayer = 0; neuronIndexInPreviousLayer < Layers[layer - 1]; neuronIndexInPreviousLayer++)
                {
                    string weight = Weights[layer - 1][neuronIndexForLayer][neuronIndexInPreviousLayer].ToString();

                    string key = $"{layer - 1}-{neuronIndexInPreviousLayer}";
                    string neuronValue = values[key];

                    if (!dictionaryEntriesWeShouldRemove.Contains(key)) dictionaryEntriesWeShouldRemove.Add(key);
                    valueFormula.Append($"{weight}*{neuronValue}+");
                }

                valueFormula.Append("");
                string value = valueFormula.ToString().Trim('+');

                values.Add($"{layer}-{neuronIndexForLayer}", $"Math.Tanh({value}+{Biases[layer - 1][neuronIndexForLayer]})");
            }

            // reduce dictionary, as each iteration embeds the previous layer
            foreach (string key in dictionaryEntriesWeShouldRemove) values.Remove(key);
        }

        string formula = $"y = {values[(Layers.Length - 1).ToString() + "-0"]}".Replace("+-", "-").Replace("++", "+");

        // app requires a method called "NonAI" that takes a double and returns a float that mimics the trained AI
        string result = "        private static float NonAI(double x)\r\n" +
                        "        {\r\n" +
                       $"            // {string.Join("-", Layers)}\r\n" +
                       $"            double y = {formula};\r\n"+
                        "\r\n" +
                        "            return (float)y;\r\n" +
                        "        }\r\n";

        // proof on Stephen's Wolfram Alpha
        File.WriteAllText($"c:\\temp\\StephenWolframSquareWaveFormula-{Id}.txt", "Paste this formula into https://www.wolframalpha.com/\r\n" + formula.Replace("Math.Tanh(", "TanH"));

        // the c# code for the formula
        File.WriteAllText($"c:\\temp\\StephenWolframSquareWaveFormula-C#-code-{Id}.txt", result);
        return result;
    }
}