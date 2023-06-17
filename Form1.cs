#define StephenWolframShape
using Microsoft.VisualBasic.Devices;
using System;
using System.Security.Cryptography;
using System.Text;

namespace WolframTest
{
    /// <summary>
    /// From https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/, he gave an example.
    /// I thought, why not create an animated one, using the coordinates he chose, and the definition of layers. 
    /// 
    /// "And what we see is that if the net is too small, it just can’t reproduce the function we want. But above some size, 
    ///  it has no problem—at least if one trains it for long enough, with enough examples. And, by the way, these pictures
    ///  illustrate a piece of neural net lore: that one can often get away with a smaller network if there’s a “squeeze” in 
    ///  the middle that forces everything to go through a smaller intermediate number of neurons." SW
    /// 
    /// This is part of the blog post: https://aimlfun.com/lore-aint-lore-unless-you-use-rigour, that will be posted in June.
    /// </summary>
    public partial class Form1 : Form
    {
        /// <summary>
        /// We use this to write the layers at the bottom of the graph.
        /// </summary>
        static readonly Font fontForLayersLabel = new("Arial", 8);

        /// <summary>
        /// This is a timer that will train the AI, and update the graph. It's inefficient, but solves various "issues". It could
        /// use multiple threads and locking - this would yield better performance. For example one thread per graph.
        /// </summary>
        private System.Windows.Forms.Timer timer;

        /// <summary>
        /// This tracks the PictureBox's we use to render the graphs.
        /// </summary>
        private PictureBox[] outputPictureBox;

        /// <summary>
        /// These are the data-points we use to train the AI.
        /// From https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/
        /// </summary>
        private readonly double[][] basicTrainingData = new double[][]
        {

#if StephenWolframShape
            /*  shape of graph:
             *        ___
             *        |  |___
             *     ___|
             */

            new double[] { -1, -1 },
            new double[] { -0.32, -1 },
            new double[] { -0.32, 1 },
            new double[] { 0.32, 1 },
            new double[] { 0.32, 0 },
            new double[] { 1, 0 }
#else // alternative shape
            new double[] { -1, -1 },
            new double[] { -0.8, -1 },
            new double[] { -0.5, -0.5 }, // extra
            new double[] { -0.32, 1 },
            new double[] { 0.32, 1 },
            new double[] { 0.32, 0 },
            new double[] { 1, 0 }
#endif
        };

        private double[][] trainingData;

        /// <summary>
        /// The data-points converted to the Bitmap's coordinate system.
        /// </summary>
        Point[] trainingPointsFromStephenWolframsArticle;

        /// <summary>
        /// The last points we plotted. We use this to avoid plotting the same points over and over again.
        /// </summary>
        readonly Dictionary<int, Point[]> lastPlottedPoints = new();

        /// <summary>
        /// Every X we redraw the graphs. This is the X. The AI doesn't learn quick enough to warrant painting every time,
        /// so this improves performance.
        /// </summary>
        readonly int frequencyOfUpdate = 10;

        /// <summary>
        /// This is the counter for the frequencyOfUpdate.
        /// </summary>
        int frequencyCounter = 0;

        /// <summary>
        /// This splits the training points into a line of "n" points. The more points, the more accurate the graph, better training.
        /// </summary>
        readonly int pointsToDivideLine = 51; // 9,11,13,15 etc work fine

        /// <summary>
        /// Demo form.
        /// </summary>
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Incorrect. They are initialised in the methods it calls!
        public Form1()
#pragma warning restore CS8618 // Non-nullable field must contain a non-null value when exiting constructor. 
        {
            InitializeComponent();

            //File.WriteAllText(@"c:\temp\output.txt", BreakDownOfNonAI());

            CreateTrainingPoints();
            CreateNeuralNetworks();

            CreatePictureBoxesToRenderGraphs();

            CreateTrainingTimer();
        }

        /// <summary>
        /// Every time this fires, it trains the AI, and every so often updates the graphs.
        /// </summary>
        private void CreateTrainingTimer()
        {
            timer = new System.Windows.Forms.Timer
            {
                Interval = 3 // 3ms
            };

            timer.Tick += new EventHandler(Timer_Tick);
            timer.Enabled = true;
        }

        /// <summary>
        /// This creates and adds the PictureBox's to the form.
        /// </summary>
        private void CreatePictureBoxesToRenderGraphs()
        {
            List<PictureBox> pictureBoxes = new();

            for (int neuralNetworkID = 0; neuralNetworkID < NeuralNetwork.s_networks.Count + 1; neuralNetworkID++)
            {
                PictureBox pictureBox = new()
                {
                    Size = new Size(200, 220),
                    SizeMode = PictureBoxSizeMode.StretchImage,
                    Image = new Bitmap(200, 220),
                    BorderStyle = BorderStyle.FixedSingle
                };

                flowLayoutPanel1.Controls.Add(pictureBox);

                pictureBoxes.Add(pictureBox);
            }

            outputPictureBox = pictureBoxes.ToArray();

            // last one is non AI
            outputPictureBox[NeuralNetwork.s_networks.Count].Image = DrawGraph(NeuralNetwork.s_networks.Count);
        }

        /// <summary>
        /// This creates each of the networks that Stephen Wolfram used in his article,
        /// from converting his pictures to layers
        /// </summary>
        private static void CreateNeuralNetworks()
        {
            NeuralNetwork.s_networks.Clear();

            // the thing worth noting is that all are input - hidden - output, and Stephen is demonstrating (1) there can be too few neurons (2) that the middle layer can be less.
            // except it doesn't.
            _ = new NeuralNetwork(0, new int[] { 1, 1 }); // <-- no hidden layers
            _ = new NeuralNetwork(1, new int[] { 1, 1, 1 }); // ?? !! 
            _ = new NeuralNetwork(2, new int[] { 1, 2, 1 }); // <-- works fine
            _ = new NeuralNetwork(3, new int[] { 1, 3, 1 });
            _ = new NeuralNetwork(4, new int[] { 1, 4, 1 });
            _ = new NeuralNetwork(5, new int[] { 1, 2, 2, 1 });
            _ = new NeuralNetwork(6, new int[] { 1, 3, 2, 1 }); // <-- squeeze middle
            _ = new NeuralNetwork(7, new int[] { 1, 3, 3, 1 });
            _ = new NeuralNetwork(8, new int[] { 1, 3, 2, 3, 1 }); // <-- squeeze middle
        }

        /// <summary>
        /// Converts the training data to the Bitmap's coordinate system. Core code from CoPilot.
        /// </summary>
        private void CreateTrainingPoints()
        {
            List<Point> pointsOnGraphToPlotTheDesiredPath = new();
            List<double[]> trainingDataXAndExpectedY = new();

            for (int i = 1; i < basicTrainingData.Length; i++)
            {
                // compute points between Point1 and basicTrainingData[i]
                double[] Point1 = basicTrainingData[i - 1];
                double[] Point2 = basicTrainingData[i];

                double x1 = Point1[0];
                double y1 = Point1[1];
                double x2 = Point2[0];
                double y2 = Point2[1];

                // compute the delta between the two points
                double xDiff = x2 - x1;
                double yDiff = y2 - y1;

                // compute the gradient for each step.
                double xStep = xDiff / pointsToDivideLine;
                double yStep = yDiff / pointsToDivideLine;

                for (int j = 0; j < pointsToDivideLine; j++)
                {
                    double x = x1 + (xStep * j);
                    double y = y1 + (yStep * j);

                    pointsOnGraphToPlotTheDesiredPath.Add(ConvertTrainingCoordinatesToPoint(new double[2] { x, y }));
                    trainingDataXAndExpectedY.Add(new double[2] { x, y });
                }
            }

            trainingPointsFromStephenWolframsArticle = pointsOnGraphToPlotTheDesiredPath.ToArray();
            trainingData = trainingDataXAndExpectedY.ToArray();
        }

        /// <summary>
        /// Used to stop the timer firing whilst it is already running.
        /// </summary>
        bool inTimer = false;

        /// <summary>
        /// When the timer fires, this trains the AI, and every so often updates the graphs.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Timer_Tick(object? sender, EventArgs e)
        {
            if (inTimer) return;

            inTimer = true;

            bool updateGraphs = (++frequencyCounter) % frequencyOfUpdate == 0;

            for (int neuralNetworkID = 0; neuralNetworkID < NeuralNetwork.s_networks.Count; neuralNetworkID++)
            {
                // backpropagate this network with the training data
                for (int trainingDataIndex = 0; trainingDataIndex < trainingData.Length; trainingDataIndex++)
                {
                    NeuralNetwork.s_networks[neuralNetworkID].BackPropagate(new double[] { trainingData[trainingDataIndex][0] }, new double[] { trainingData[trainingDataIndex][1] });
                }

                if (updateGraphs)
                {
                    outputPictureBox[neuralNetworkID].Image?.Dispose();
                    outputPictureBox[neuralNetworkID].Image = DrawGraph(neuralNetworkID);

                    // we're plotting a number of networks, we only need to update the label for the first one
                    if (neuralNetworkID == 0) Text = $"{frequencyCounter}";
                }
            }

            inTimer = false;
        }

        /// <summary>
        /// Draws the graph for the specified network.
        /// </summary>
        /// <param name="networkId"></param>
        /// <returns></returns>
        private Bitmap DrawGraph(int networkId)
        {
            Point centre = ConvertTrainingCoordinatesToPoint(new double[] { 0, 0 });

            Bitmap graphBitmap = new(200, 220); // 200x200 + 20 for the layers label

            using Graphics graphicsForGraphBitmap = Graphics.FromImage(graphBitmap);
            graphicsForGraphBitmap.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            graphicsForGraphBitmap.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;

            graphicsForGraphBitmap.Clear(Color.White);
            DrawAxisThruCentreOfGraph(centre, graphicsForGraphBitmap);
            DrawDottedRedLineIndicatingTheValuesTheAIisExpectedToLearn(graphicsForGraphBitmap);

            List<Point> points = GetPointsPredictedByAI(networkId);

            // draw the last plotted points in silver, this enables us to see how the line moved since last time
            if (lastPlottedPoints.ContainsKey(networkId)) graphicsForGraphBitmap.DrawLines(Pens.Silver, lastPlottedPoints[networkId]);

            // draw the current line in blue
            graphicsForGraphBitmap.DrawLines(Pens.Blue, points.ToArray());

            // store the last plotted points for this network
            if (lastPlottedPoints.ContainsKey(networkId))
                lastPlottedPoints[networkId] = points.ToArray();
            else
                lastPlottedPoints.Add(networkId, points.ToArray());

            LabelGraphWithLayers(networkId, graphicsForGraphBitmap);

            return graphBitmap;
        }

        /// <summary>
        /// Draw a "+" shape at the centre of the graph.
        /// </summary>
        /// <param name="centre"></param>
        /// <param name="graphicsForGraphBitmap"></param>
        private static void DrawAxisThruCentreOfGraph(Point centre, Graphics graphicsForGraphBitmap)
        {
            // draw axis (center X vertical and horizontal)
            graphicsForGraphBitmap.DrawLine(Pens.DarkGray, 0, centre.Y, 199, centre.Y);
            graphicsForGraphBitmap.DrawLine(Pens.DarkGray, centre.X, 0, centre.X, 199);
        }

        /// <summary>
        /// Draw red dotted line showing the expected path for the AI to mimic.
        /// </summary>
        /// <param name="graphicsForGraphBitmap"></param>
        private void DrawDottedRedLineIndicatingTheValuesTheAIisExpectedToLearn(Graphics graphicsForGraphBitmap)
        {
            foreach (Point p in trainingPointsFromStephenWolframsArticle)
            {
                graphicsForGraphBitmap.FillEllipse(Brushes.Red, p.X - 1, p.Y - 1, 2, 2);
            }
        }

        /// <summary>
        /// Add the label to the bottom of the graph to show the layers.
        /// </summary>
        /// <param name="networkId"></param>
        /// <param name="graphicsForGraphBitmap"></param>
        private void LabelGraphWithLayers(int networkId, Graphics graphicsForGraphBitmap)
        {
            // draw Layers array at the center bottom of bitmap
            string layers = networkId == NeuralNetwork.s_networks.Count ? "NON-AI" : string.Join(",", NeuralNetwork.s_networks[networkId].Layers);
            Size size = TextRenderer.MeasureText(layers, fontForLayersLabel);
            graphicsForGraphBitmap.DrawString(layers, fontForLayersLabel, Brushes.Black, new PointF(100 - size.Width / 2, 220 - size.Height));
        }

        /// <summary>
        /// Ask the AI to get the points to plot.
        /// We're training it on a few points, but we want to see how it does on all points.
        /// </summary>
        /// <param name="networkId"></param>
        /// <returns></returns>
        private static List<Point> GetPointsPredictedByAI(int networkId)
        {
            List<Point> points = new();
            bool nonAI = (NeuralNetwork.s_networks.Count == networkId);

            for (float x = -1; x < 1; x += 0.05f)
            {
                float y = nonAI ? NonAI(x) : (float)NeuralNetwork.s_networks[networkId].FeedForward(new double[] { x })[0];

                points.Add(ConvertTrainingCoordinatesToPoint(new double[] { x, y }));
            }

            return points;
        }

        /// <summary>
        /// Here's the same thing without using AI. The following SAVES the code.
        ///         _ = NeuralNetwork.s_networks[neuralNetworkID].Formula(); 
        /// Paste the code on top of the below.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private static float NonAI(double x)
        {
            double w1 = 31.546993122145782;
            double b1 = -7.9125549106709085;
            double b2 = -14.220176897512127;
            double w2 = -46.89539279833847;
            double w3 = -9.733774606758697;
            double w4 = -10.866331730849167;
            double bOut = -1.132557161957376;

            double y = Math.Tanh(w3 * Math.Tanh(w1 * x + b1) + w4 * Math.Tanh(w2 * x + b2) + bOut);

            return (float)y;
        }

        /// <summary>
        /// Maps training data (-1..1,-1..1) to the Bitmap's coordinate system, on a 200x200 grid.
        /// </summary>
        /// <param name="trainingData"></param>
        /// <returns></returns>
        static Point ConvertTrainingCoordinatesToPoint(double[] trainingData)
        {
            return new Point((int)Math.Round(trainingData[0] * 99 + 99), (int)Math.Round(199 - (trainingData[1] * 99 + 99)));
        }

        /// <summary>
        /// Allow user to "pause", "save" and "write formula" for the neural networks.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form1_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                // pause the learning
                case Keys.P:
                    timer.Enabled = !timer.Enabled;
                    break;

                // save the trained model
                case Keys.S:
                    // stop it learning
                    timer.Enabled = false;

                    // wait for timer to finish
                    while (inTimer) Application.DoEvents();

                    NeuralNetwork.SaveTrainedModel();

                    // continue learning
                    timer.Enabled = true;
                    break;

                // write the "formula" for the neural network                
                case Keys.F:
                    // stop it learning
                    timer.Enabled = false;

                    // wait for timer to finish
                    while (inTimer) Application.DoEvents();

                    // output the formula for each network
                    foreach (int neuralNetworkID in NeuralNetwork.s_networks.Keys)
                        _ = NeuralNetwork.s_networks[neuralNetworkID].Formula();

                    // continue learning
                    timer.Enabled = true;
                    break;
            }
        }
    }
}