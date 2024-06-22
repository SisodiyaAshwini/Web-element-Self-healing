using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AP_SelfHealing.MachineLearning.DataModels
{
    public class PredictedElement
    {
        [ColumnName("PredictedLabel")]
        public string? Element;
    }
}
