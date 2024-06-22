using Microsoft.ML.Data;

namespace AP_SelfHealing.MachineLearning.Common
{
    public interface ITrainerBase
    {
        string Name { get; }
        void Fit(string trainingFileName);
        RegressionMetrics Evaluate();
        void Save();
    }
}
