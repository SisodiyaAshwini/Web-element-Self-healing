using System.Data.SqlClient;
using System.Runtime.InteropServices;

namespace AP_SelfHealing
{
    public static class DataConnection
    {
        public static void Connection()
        { // Connection string
            string connectionString = @"Server=localhost\SQLEXPRESS;Database=MLData;Integrated Security=True;";

            // Prompt the user to enter a URL
            Console.WriteLine("Enter the URL:");
            string url = Console.ReadLine();
            // Establish connection
            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                try
                {
                    connection.Open();
                    Console.WriteLine("Connected to SQL Server database");

                    // Execute SQL queries here
                    string insertQuery = @"INSERT INTO PageData 
                    ([URL],[Element], Tag, XPath, ControlId, [Name], CSSClass, [Value], [Label], [Role], [Type], TabIndex, [Placeholder], Line, LinePosition, Href, [Title]) 
                    VALUES (@URL, @Element, @Tag, @XPath, @ControlId, @Name, @CSSClass, @Value, @Label, @Role, @Type, @TabIndex, @Placeholder, @Line, @LinePosition, @Href, @Title)";

                    var controls = ReadDocument.GetControlAttributes(url);
                    foreach (var control in controls)
                    {
                        // Create a SqlCommand object
                        using (SqlCommand command = new SqlCommand(insertQuery, connection))
                        {
                            var attributes = control.GetAttributes(new string[11] { "name", "class", "value", "role", "tabindex", "type", "aria-label", "role", "placeholder", "href", "title" });
                            var attlst = attributes.ToList();
                            string title = attlst.Where(x => x != null && x.Name == "title").Count() == 1 ? attlst.Where(x => x != null && x.Name == "title").ToList().First().Value : string.Empty;
                            string type = attlst.Where(x => x != null && x.Name == "type").Count() == 1 ? attlst.Where(x => x != null && x.Name == "type").ToList().First().Value:string.Empty;
                            string name = attlst.Where(x => x != null && x.Name == "name").Count() == 1 ? attlst.Where(x => x != null && x.Name == "name").ToList().First().Value : string.Empty;
                            command.Parameters.AddWithValue("@URL"
                                , url);
                            command.Parameters.AddWithValue("@Element", string.IsNullOrEmpty(name) 
                                ? (string.IsNullOrEmpty(type)
                                ?(string.IsNullOrEmpty(title)?DBNull.Value:title) :type) : name);
                                
                            command.Parameters.AddWithValue("@Tag", control.Name);
                            command.Parameters.AddWithValue("@XPath", control.XPath);
                            
                            command.Parameters.AddWithValue("@ControlId", string.IsNullOrEmpty(control.Id) ? DBNull.Value : control.Id);
                            
                            command.Parameters.AddWithValue("@Name", string.IsNullOrEmpty(name)? DBNull.Value:name);
                            //command.Parameters.AddWithValue("@Name", (attlst.Where(x => x != null && x.Name == "name").Count() == 1) ? attlst.Where(x => x != null && x.Name == "name").ToList().First().Value : DBNull.Value);
                            command.Parameters.AddWithValue("@CSSClass", (attlst.Where(x => x != null && x.Name == "class").Count() == 1) ? attlst.Where(x => x != null && x.Name == "class").ToList().First().Value : DBNull.Value);
                            command.Parameters.AddWithValue("@Value", (attlst.Where(x => x != null && x.Name == "value").Count() == 1) ? attlst.Where(x => x != null && x.Name == "value").ToList().First().Value : DBNull.Value);
                            command.Parameters.AddWithValue("@Label", (attlst.Where(x => x != null && x.Name == "aria-label").Count() == 1) ? attlst.Where(x => x != null && x.Name == "aria-label").ToList().First().Value : DBNull.Value);
                            command.Parameters.AddWithValue("@Role", (attlst.Where(x => x != null && x.Name == "role").Count() == 1) ? attlst.Where(x => x != null && x.Name == "role").ToList().First().Value : DBNull.Value);
                            //command.Parameters.AddWithValue("@Type", (attlst.Where(x => x != null && x.Name == "type").Count() == 1) ? attlst.Where(x => x != null && x.Name == "type").ToList().First().Value : DBNull.Value);
                            command.Parameters.AddWithValue("@Type", string.IsNullOrEmpty(type)? DBNull.Value:type);
                            command.Parameters.AddWithValue("@TabIndex", (attlst.Where(x => x != null && x.Name == "tabindex").Count() == 1) ? attlst.Where(x => x != null && x.Name == "tabindex").ToList().First().Value : DBNull.Value);
                            command.Parameters.AddWithValue("@Placeholder", (attlst.Where(x => x != null && x.Name == "placeholder").Count() == 1) ? attlst.Where(x => x != null && x.Name == "placeholder").ToList().First().Value : DBNull.Value);
                            //command.Parameters.AddWithValue("@Title", (attlst.Where(x => x != null && x.Name == "title").Count() == 1) ? attlst.Where(x => x != null && x.Name == "title").ToList().First().Value : DBNull.Value);
                            command.Parameters.AddWithValue("@Title", string.IsNullOrEmpty(title)? DBNull.Value:title);
                            command.Parameters.AddWithValue("@Href", (attlst.Where(x => x != null && x.Name == "href").Count() == 1) ? attlst.Where(x => x != null && x.Name == "href").ToList().First().Value : DBNull.Value);
                            //    }
                            //}
                            // Add parameters to the query
                            //command.Parameters.AddWithValue("@Name", control.GetAttributeValue("name", DBNull.Value));
                            //command.Parameters.AddWithValue("@CSSClass", control.GetAttributeValue("class", DBNull.Value));
                            //command.Parameters.AddWithValue("@value", control.GetAttributeValue("value", DBNull.Value));

                            command.Parameters.AddWithValue("@Line", control.Line);
                            command.Parameters.AddWithValue("@LinePosition", control.LinePosition);


                            // Execute the query
                            int rowsAffected = command.ExecuteNonQuery();
                            // Check if the data was inserted successfully
                            if (rowsAffected > 0)
                            {
                                Console.WriteLine("Data inserted successfully.");
                            }
                            else
                            {
                                Console.WriteLine("No rows were affected. Data insertion failed.");
                            }
                        }
                    }
                }
                catch (SqlException ex)
                {
                    Console.WriteLine("Error connecting to SQL Server database: " + ex.Message);
                }
            }
        }
    }
}
        