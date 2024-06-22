using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AP_SelfHealing.MachineLearning.DataModels
{
    public class Model
    {
        [LoadColumn(0)]
        public string Element { get; set; }
        [LoadColumn(1)]
        public string? Line { get; set; }
        [LoadColumn(2)]
        public string? LinePosition { get; set; }
        [LoadColumn(3)]
        public string? ControlId { get; set; }
        [LoadColumn(4)]
        public string? Name { get; set; }
        [LoadColumn(5)]
        public string? CCSClass { get; set; }
        [LoadColumn(6)]
        public string? Value { get; set; }

        [LoadColumn(7)]
        public string? Role { get; set; }
        [LoadColumn(8)]
        public string? Type { get; set; }
        [LoadColumn(9)]
        public string? TabIndex { get; set; }
        [LoadColumn(10)]
        public string? Placeholder { get; set; }
        [LoadColumn(11)]
        public string? Title { get; set; }
        [LoadColumn(12)]
        public string? Href { get; set; }

    }

    public class TransformedData : Model3
    {
        public string? Line { get; set; }
        public string? LinePosition { get; set; }
        public string? ControlId { get; set; }
        public string? Name { get; set; }
        public string? CCSClass { get; set; }
        public string? Value { get; set; }
        public string? Role { get; set; }
        public string? Type { get; set; }
        public string? TabIndex { get; set; }
        public string? Placeholder { get; set; }
        public string? Title { get; set; }
        public string? Href { get; set; }
        public float Label { get; set; }
        //public string[] Features { get; set; }
        //public float[] FeaturesNum { get; set; }
        //public string[] OutputTokens { get; set; }
    }
}
