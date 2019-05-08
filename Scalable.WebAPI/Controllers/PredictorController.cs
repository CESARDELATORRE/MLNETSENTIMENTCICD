using System;
using Microsoft.Extensions.ML;
using Microsoft.AspNetCore.Mvc;
using SentimentModel.Model.DataModels;

namespace Scalable.WebAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class PredictorController : ControllerBase
    {
        private readonly PredictionEnginePool<ModelInput, ModelOutput> _predictionEnginePool;

        public PredictorController(PredictionEnginePool<ModelInput, ModelOutput> predictionEnginePool)
        {
            // Get the ML Model Engine injected, for scoring
            _predictionEnginePool = predictionEnginePool;
        }

        // GET api/predictor/sentimentprediction?sentimentText=ML.NET is awesome!
        [HttpGet]
        [Route("sentimentprediction")]
        public ActionResult<string> PredictSentiment([FromQuery]string sentimentText)
        {
            ModelInput sampleData = new ModelInput() { Text = sentimentText };

            //Predict sentiment
            ModelOutput prediction = _predictionEnginePool.Predict(sampleData);

            bool isPositiveSentiment = prediction.PositiveSentiment;
            float probability = CalculatePercentage(prediction.Score);
            string retVal = $"Prediction Positive Sentiment: '{isPositiveSentiment.ToString()}' with {probability.ToString()}% probability of being positive for the text '{sentimentText}'";

            return retVal;

        }

        public static float CalculatePercentage(double value)
        {
            return 100 * (1.0f / (1.0f + (float)Math.Exp(-value)));
        }
    }
}