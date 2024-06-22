using System.Text;
using HtmlAgilityPack;

namespace AP_SelfHealing
{
    public static class ReadDocument
    {
        public static List<HtmlNode> GetControlAttributes(string url)
        {
            {                
                try
                {
                    HtmlWeb web = new HtmlWeb();
                    web.OverrideEncoding = Encoding.UTF8;
                    HtmlDocument doc = web.Load(url);
                    HtmlNode.ElementsFlags.Remove("form");

                    //Get all HTML controls
                    var controls = doc.DocumentNode.Descendants("input")
                                            .Concat(doc.DocumentNode.Descendants("select"))
                                            .Concat(doc.DocumentNode.Descendants("textarea"))
                                            .Concat(doc.DocumentNode.Descendants("button"))
                                            .Concat(doc.DocumentNode.Descendants("a")).ToList();

                    //var controls = doc.DocumentNode.Descendants("input").ToList();

                    //Print details of each control
                    //foreach (var control in controls)
                    //{
                    //    Console.WriteLine($"Control Tag: {control.Name}"); //OrignalName
                    //    Console.WriteLine($"Control XPath: {control.XPath}");
                    //    Console.WriteLine($"Control ID: {control.GetAttributeValue("id", "No ID attribute")}");
                    //    Console.WriteLine($"Control Name: {control.GetAttributeValue("name", "No name attribute")}");
                    //    Console.WriteLine($"Control CSSClass: {control.GetAttributeValue("class", "No class attribute")}");
                    //    Console.WriteLine($"Control Value: {control.GetAttributeValue("value", "No value attribute")}");
                    //    Console.WriteLine($"Control Style: {control.GetAttributeValue("style", "No style attribute")}");
                    //    Console.WriteLine();
                    //}
                    return controls;
                }
                catch (Exception ex)
                { 
                    Console.WriteLine($"Error: {ex.Message}");
                    return null;
                }
            }
        }
    }
}
