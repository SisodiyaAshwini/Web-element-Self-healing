using AP_SelfHealing;
using AP_SelfHealing.MachineLearning.DataModels;

public class Program
{
    public static void Main(string []args)
    {
        //DataConnection.Connection();

        ElementPrediction elementPrediction = new ElementPrediction();
        elementPrediction.LoadAndPrepareData();

        //ElementPrediction4.LoadAndPrepareData();
        //var newSample = new Model
        //{
        //    Element = "Password",
        //    Line = "639",
        //    LinePosition = "0",
        //    ControlId = "pass",
        //    Name = "Password",
        //    CCSClass = "selectors-input jsSelector",
        //    Value = "",
        //    Label =null,
        //    Role = null,
        //    Type = "password",
        //    TabIndex = null,
        //    Placeholder = "Enter Password",
        //    Title="Password",
        //    Href=null
        //};



        //var trainers = new List<ITrainerBase>
        //    {
        //        new RandomForestTrainer(2, 5),
        //        new RandomForestTrainer(5, 10),
        //        new RandomForestTrainer(10, 20)
        //    };

        //trainers.ForEach(t => TrainEvaluatePredict(t, newSample));
    }

    //static void TrainEvaluatePredict(ITrainerBase trainer, Model newSample)
    //{
    //    Console.WriteLine("*******************************");
    //    Console.WriteLine($"{trainer.Name}");
    //    Console.WriteLine("*******************************");

    //    trainer.Fit(".\\Data\\boston_housing_dataset.csv");

    //    var modelMetrics = trainer.Evaluate();

    //    Console.WriteLine($"Loss Function: {modelMetrics.LossFunction:0.##}{Environment.NewLine}" +
    //                      $"Mean Absolute Error: {modelMetrics.MeanAbsoluteError:#.##}{Environment.NewLine}" +
    //                      $"Mean Squared Error: {modelMetrics.MeanSquaredError:#.##}{Environment.NewLine}" +
    //                      $"RSquared: {modelMetrics.RSquared:0.##}{Environment.NewLine}" +
    //                      $"Root Mean Squared Error: {modelMetrics.RootMeanSquaredError:#.##}");

    //    trainer.Save();

    //    var predictor = new Predictor();
    //    var prediction = predictor.Predict(newSample);
    //    Console.WriteLine("------------------------------");
    //    Console.WriteLine($"Prediction: {prediction.Element:#.##}");
    //    Console.WriteLine("------------------------------");
   
    //}
    
}

    