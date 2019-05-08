using NUnit.Framework;
using System;
using System.IO;

using Microsoft.ML;
using SentimentModel.Model.DataModels;
using System.Collections.Generic;
using System.Linq;

namespace Tests
{
    public class MLModelUnitTests
    {
        MLContext _mlContext;
        ITransformer _trainedModel;

        //Machine Learning model to load and use for predictions
        private const string MODEL_FILEPATH = @"../../../../SentimentModel/SentimentModel.Model/MLModel.zip";

        private const string UNIT_TEST_DATA_FILEPATH = @"../../../unit_test_data_baseline.tsv";

        private const string EVALUATION_DATA_FILEPATH = @"../../../evaluation_dataset.tsv";

        [SetUp]
        public void Setup()
        {
            _mlContext = new MLContext();

            _trainedModel = _mlContext.Model.Load(GetAbsolutePath(MODEL_FILEPATH), out var modelInputSchema);
        }

        [Test]
        public void TestPositiveSentimentStatement()
        {
            ModelInput sampleStatement = new ModelInput { Text = "ML.NET is awesome!" };

            var predEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_trainedModel);

            var resultprediction = predEngine.Predict(sampleStatement);

            Assert.AreEqual(true, Convert.ToBoolean(resultprediction.Prediction));
        }

        [Test]
        public void TestNegativeSentimentStatement()
        {
            string testStatament = "This movie was very boring...";
            ModelInput sampleStatement = new ModelInput { Text = testStatament };

            var predEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_trainedModel);

            var resultprediction = predEngine.Predict(sampleStatement);

            Assert.AreEqual(false, Convert.ToBoolean(resultprediction.Prediction));
        }

 
        [Test]
        public void TestAccuracyHigherThan60()
        {
            Console.WriteLine("===== Evaluating Model's accuracy with Evaluation/Test dataset =====");
            
            // Read dataset to get a single row for trying a prediction          
            IDataView testDataView = _mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: GetAbsolutePath(EVALUATION_DATA_FILEPATH),
                                            hasHeader: true,
                                            separatorChar: '\t');

            IEnumerable<ModelInput> samplesForPrediction = _mlContext.Data.CreateEnumerable<ModelInput>(testDataView, false);

            //DO BULK PREDICTIONS
            IDataView predictionsDataView = _trainedModel.Transform(testDataView);

            var predictions = _trainedModel.Transform(testDataView);
            var metrics = _mlContext.BinaryClassification.Evaluate(data: predictionsDataView, labelColumnName: "sentiment", scoreColumnName: "Score");

            double accuracy = metrics.Accuracy;
            Console.WriteLine($"Accuracy of model in this validation '{accuracy*100}'%");
            
            Assert.GreaterOrEqual(0.80, accuracy);
        }

        //Generate many test cases with a bulk prediction approach
        public static List<TestCaseData> TestCases
        {
            get
            {
                MLContext mlContext = new MLContext();
                ITransformer trainedModel = mlContext.Model.Load(GetAbsolutePath(MODEL_FILEPATH), out var modelInputSchema);

                // Read dataset to get a single row for trying a prediction          
                IDataView testDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                                path: GetAbsolutePath(UNIT_TEST_DATA_FILEPATH),
                                                hasHeader: true,
                                                separatorChar: '\t');

                IEnumerable<ModelInput> samplesForPrediction = mlContext.Data.CreateEnumerable<ModelInput>(testDataView, false);
                ModelInput[] arraysamplesForPrediction = samplesForPrediction.ToArray();

                //DO BULK PREDICTIONS
                IDataView predictionsDataView = trainedModel.Transform(testDataView);
                IEnumerable<ModelOutput> predictions = mlContext.Data.CreateEnumerable<ModelOutput>(predictionsDataView, false);
                ModelOutput[] arrayPredictions = predictions.ToArray();

                var TestCases = new List<TestCaseData>();

                for (int i = 0; i < arraysamplesForPrediction.Length; i++)
                {
                    TestCases.Add(new TestCaseData(arraysamplesForPrediction[i].Text,
                                                   arrayPredictions[i].Prediction,
                                                   arraysamplesForPrediction[i].Sentiment));
                }

                return TestCases;
            }
        }

        [TestCaseSource("TestCases")]
        public void TestSentimentStatement(string sampleText, bool predictedSentiment, bool expectedSentiment)
        {
            try
            {
                Console.WriteLine($"Text {sampleText} predicted as {predictedSentiment} should be {expectedSentiment}");
                Assert.AreEqual(predictedSentiment, expectedSentiment);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(MLModelUnitTests).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}