using AP_SelfHealing.MachineLearning.Common;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML;

namespace AP_SelfHealing.MachineLearning.Trainers
{
    /// <summary>
    /// Class that uses Random Forest algorithm.
    /// </summary>
    public sealed class RandomForestTrainer : TrainerBase<FastForestRegressionModelParameters>
    {
        public RandomForestTrainer(int numberOfLeaves, int numberOfTrees) : base()
        {
            Name = $"Random Forest-{numberOfLeaves}-{numberOfTrees}";
            _model = MlContext.Regression.Trainers.FastForest(numberOfLeaves: numberOfLeaves,
                                      numberOfTrees: numberOfTrees);
        }
    }
}
