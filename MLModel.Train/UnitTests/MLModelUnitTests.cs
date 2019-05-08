using NUnit.Framework;
using System;
using System.IO;

using Microsoft.ML;
using SentimentModel.Model.DataModels;

namespace Tests
{
    public class MLModelUnitTests
    {
        MLContext _mlContext;
        ITransformer _trainedModel;

        //Machine Learning model to load and use for predictions
        private const string MODEL_FILEPATH = @"../../../../SentimentModel/SentimentModel.Model/MLModel.zip";

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

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(MLModelUnitTests).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}