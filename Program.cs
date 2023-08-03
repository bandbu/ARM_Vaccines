using Accord.IO;
using Accord.Math;
using System.Data;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML;
using Microsoft.ML.Data;
using static AssociationRuleMining.Program.QACoPilotZero;
using NewtonsoftJson = Newtonsoft.Json;

namespace AssociationRuleMining
{
    class Program
    {
        static void Main(string[] args)
        {
            Menu();
        }
        
        static void Menu()
        {
            bool inMenu = true;
            string dataPath = "C:\\Khoa\\Git\\Association-Rule-Mining\\data2.csv";
            
            string rulesPath = "C:\\Khoa\\Git\\Association-Rule-Mining\\Rules.json";
            string rulesClusterPath = "C:\\Khoa\\Git\\Association-Rule-Mining\\RulesClusterIncluded.json";
            string TypeDictPath = "C:\\Khoa\\Git\\Association-Rule-Mining\\TypeDict.json";
            List<AssociationRule> rules_ = new List<AssociationRule>();
            while (inMenu)
            {
                Console.WriteLine("============== QA Co-pilot Zero ===============");
                Console.WriteLine("1. Train Co-pilot");
                Console.WriteLine("2. Load and run Train Co-pilot");
                if (rules_.Count > 0)
                {
                    Console.WriteLine("3. Print all rules");
                    Console.WriteLine("4. Clustering Rules");
                }
                Console.WriteLine("0. Exit");

                Console.WriteLine("Choose an option:");
                int choice = Convert.ToInt16(Console.ReadLine());
                switch (choice)
                {
                    case 0: inMenu = false; break;
                    case 1:
                        {
                            #region Trainning Session
                            List<List<string>> transactions = QACoPilotZero.DataReading(dataPath);
                            Console.WriteLine("Data read");
                            float minSupport = 0;
                            float minConfidence = 0;
                            int suportControl = 10;

                            //Tính toán luật kết hợp
                            Console.WriteLine("Begin Trainning");
                            List<AssociationRule> rules = QACoPilotZero.GenerateAssociationRules(transactions, minSupport, minConfidence, suportControl);
                            rules_ = rules.ToList();
                            Console.WriteLine("______________________________________________");
                            Console.WriteLine(">> Begin Write To File");
                            QACoPilotZero.WriteToFile(rulesPath, rules);

                            Stopwatch stopwatch = new Stopwatch();
                            Console.WriteLine("5.Begin clustering");
                            stopwatch = new Stopwatch();
                            stopwatch.Start();

                            var mlContext = new MLContext(seed: 0);
                            string featuresColumnName = "Features";
                            var schemaDefinition = SchemaDefinition.Create(typeof(AssociationRuleNumber));
                            var pipeline = mlContext.Transforms
                                .Concatenate(featuresColumnName, "Support", "Confidence")
                                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 5));
                            List<AssociationRuleNumber> trainnindata = new List<AssociationRuleNumber>();
                            foreach (var rule in rules_)
                            {
                                trainnindata.Add(new AssociationRuleNumber(rule.Support, rule.Confidence));
                            }
                            IDataView dataView = mlContext.Data.LoadFromEnumerable(trainnindata, schemaDefinition);
                            var model = pipeline.Fit(dataView);
                            Console.WriteLine(">>> " + rules_.Count + " Done in: " + stopwatch.Elapsed);
                            Console.WriteLine("6.Finalize a data");
                            stopwatch = new Stopwatch();
                            stopwatch.Start();
                            var predictor = mlContext.Model.CreatePredictionEngine<AssociationRuleNumber, ClusterPrediction>(model);
                            for (int t = 0; t < rules_.Count; t++)
                            {
                                rules_[t].Cluster = predictor.Predict(new AssociationRuleNumber(rules_[t].Support, rules_[t].Confidence)).PredictedClusterId;
                            }
                            QACoPilotZero.WriteToFile(rulesClusterPath, rules_);
                            Console.WriteLine("7. Generate TypeDict");
                            var kMeansModel = model.LastTransformer.Model;

                            // Lấy ra các điểm trung tâm của các cluster
                            VBuffer<float>[] centroids = default;
                            model.LastTransformer.Model.GetClusterCentroids(ref centroids, out int k);
                            Dictionary<int, float> typedict_raw = new Dictionary<int, float>();
                            for(int j=0;j<centroids.Length;j++)
                            {
                                typedict_raw.Add(j,(float)Math.Sqrt((centroids[j].GetValues()[0] * centroids[j].GetValues()[0]) + (centroids[j].GetValues()[1]* centroids[j].GetValues()[1])));
                            }
                            //var sortedDict = typedict_raw.OrderBy(entry => entry.Value);
                            var sortedDict = typedict_raw.OrderBy(entry => entry.Value)
                             .ToDictionary(pair => pair.Key, pair => pair.Value);
                            Dictionary<int, int> typedict = new Dictionary<int, int>();
                            int i = 1;
                            foreach(var item  in sortedDict)
                            {
                                typedict.Add((item.Key+1),i);
                                i++;
                            }
                            string json = JsonSerializer.Serialize(typedict);

                            // Gỡ lỗi: Kiểm tra số lượng phần tử trong danh sách


                            try
                            {
                                using (StreamWriter writer = new StreamWriter(TypeDictPath))
                                {
                                    writer.Write(json);
                                }

                                Console.WriteLine("- File written successfully");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine("- Exception message: " + ex.Message);
                            }
                            Console.WriteLine("All Done!");
                            #endregion
                        }
                        break;
                    case 2:
                        {
                            #region Using Session
                            List<AssociationRule> rules2 = QACoPilotZero.ReadFromFile(rulesClusterPath);
                            Dictionary<float, float> TypeDict = new Dictionary<float, float>();
                            string json = File.ReadAllText(TypeDictPath);
                            Dictionary<float, float>? dictionary = JsonSerializer.Deserialize<Dictionary<float, float>>(json);
                            TypeDict = dictionary;
                            TypeDict.Add(0,0);
                            rules_ = rules2.ToList();
                            List<string> inputItem = new List<string> { "0", "U6", "40" };
                            String NextItem = "11";
                            var result = QACoPilotZero.AvailableChecking(inputItem, NextItem, rules2, TypeDict);
                            Console.WriteLine("(" + string.Join(", ", inputItem) + ") + " + NextItem + " (Confidence="+result[0]+", type = " + result[1] +")");
                            #endregion
                        }
                        break;
                    case 3:
                        {
                            var sorted_rule = rules_.OrderBy(item => item.Confidence);
                            foreach (var rule in sorted_rule)
                            {
                                Console.WriteLine(rule.ToString());
                            }
                            Console.WriteLine("---------------------------------");
                        }
                        break;
                }
                Console.ReadLine();
                Console.Clear();
            }
        }


        public class AssociationRule
        {
            public List<string> Antecedent { get; }
            public List<string> Consequent { get; }
            public float Support { get; }
            public float Confidence { get; }
            public float Count { get; }

            private uint cluster;
            public uint Cluster
            {
                get
                {
                    return cluster; // Trả về giá trị của thuộc tính khi được truy cập
                }
                set
                {
                    cluster = value; // Thiết lập giá trị mới cho thuộc tính khi gán giá trị
                }
            }

            public AssociationRule(List<string> antecedent, List<string> consequent, float support, float confidence,float count)
            {
                Antecedent = antecedent;
                Consequent = consequent;
                Support = support;
                Confidence = confidence;
                Count = count;
                cluster=0;
            }


            [JsonConstructor]
            public AssociationRule(List<string> Antecedent, List<string> Consequent, float Support, float Confidence, float Count, uint cluster)
            {
                this.Antecedent = Antecedent;
                this.Consequent = Consequent;
                this.Support = Support;
                this.Confidence = Confidence;
                this.Count = Count;
                this.cluster = cluster;
            }

            public override string ToString()
            {
                string antecedentString = string.Join(", ", Antecedent);
                string consequentString = string.Join(", ", Consequent);

                return $"{antecedentString} => {consequentString} (support: {Math.Round(Support * 100, 2)}%, confidence: {Math.Round(Confidence * 100, 2)}%, count: {Count}, cluster:{cluster})";
            }
        }

        public class AssociationRuleNumber
        {
            public float Support { get; }
            public float Confidence { get; }
            public AssociationRuleNumber(float support, float confidence)
            {
                Support = support;
                Confidence = confidence;
            }  
        }

        public static class QACoPilotZero
        {
            static List<List<string>> VaccineOnly(List<List<string>> transactions)
            {
                List<List<string>> transactions_raw = transactions.DeepClone();
                List<List<string>> transactions_ = new List<List<string>>();
                foreach (var trans in transactions_raw)
                {
                    trans.RemoveAt(0);
                    trans.RemoveAt(0);
                    transactions_.Add(trans);
                }
                return transactions_;
            }

            //-----------------------------------
            #region subsets gen
            static void GenerateSubsetsHelper(List<List<string>> transactions, List<string> items, int k, List<string> currentSubset, List<List<string>> subsets)
            {
                if (k == 0)
                {
                    if (IsSubsetPresent(transactions, currentSubset))
                    {
                        subsets.Add(new List<string>(currentSubset));
                    }
                    return;
                }

                int n = items.Count;

                for (int i = 0; i < n; i++)
                {
                    currentSubset.Add(items[i]);

                    GenerateSubsetsHelper(transactions, items.Skip(i + 1).ToList(), k - 1, currentSubset, subsets);

                    currentSubset.RemoveAt(currentSubset.Count - 1);
                }
            }

            static bool IsSubsetPresent(List<List<string>> transactions, List<string> subset)
            {
                foreach (var transaction in transactions)
                {
                    if (subset.All(item => transaction.Contains(item)))
                    {
                        return true;
                    }
                }
                return false;
            }
            //-----------------------------------
            #endregion
            public static void WriteToFile(string filePath, List<AssociationRule> rules)
            {
                string json = JsonSerializer.Serialize(rules);

                // Gỡ lỗi: Kiểm tra số lượng phần tử trong danh sách
                if (rules.Count == 0)
                {
                    Console.WriteLine("- Rules list is null");
                    return;
                }

                try
                {
                    using (StreamWriter writer = new StreamWriter(filePath))
                    {
                        writer.Write(json);
                    }

                    Console.WriteLine("- File written successfully");
                }
                catch (Exception ex)
                {
                    Console.WriteLine("- Exception message: " + ex.Message);
                }
            }

            public static List<AssociationRule> ReadFromFile(string filePath)
            {
                using (StreamReader reader = new StreamReader(filePath))
                {
                    string json = reader.ReadToEnd();

                    List<AssociationRule> rules = JsonSerializer.Deserialize<List<AssociationRule>>(json);

                    return rules;
                }
            }

            public static List<List<String>> DataReading(String filePath)
            {
                List<List<string>> data = new List<List<string>>();

                using (StreamReader reader = new StreamReader(filePath))
                {
                    string line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        List<string> row = line.Split(',').ToList();
                        // Xử lý cột cuối có các chuỗi được phân tách bằng dấu phẩy
                        string lastColumn = row.Last();
                        row.RemoveAt(row.Count - 1);
                        row.AddRange(lastColumn.Split(','));
                        //row.RemoveAt(2);
                        //row.RemoveAt(1); //tạm thời bỏ ra vì chưa xử lý tuổi
                        if (Convert.ToInt32(row[1]) < 15) row[1] = "NB";
                        else if (Convert.ToInt32(row[1]) < 61) row[1] = "LC";
                        else if (Convert.ToInt32(row[1]) < 2195) row[1] = "U6";
                        else if (Convert.ToInt32(row[1]) < 6575) row[1] = "U18";
                        else if (Convert.ToInt32(row[1]) < 17155) row[1] = "U47";
                        else row[1] = "OLD";
                        for (int i = 0; i < row.Count; i++)
                        {
                            string[] elements = row[i].Split('-');
                            row[i] = elements[0];
                        }
                        data.Add(row);
                    }
                }

                return data;
            }

            //-------------------------------------------------------------------------------------------
            public static float[] AvailableChecking(List<String> InputData, String NextItem, List<AssociationRule> rules,Dictionary<float,float> TypeDict)
            {
                Console.WriteLine("--------------");
                if (InputData.Count < 3)
                {
                    float[] resultx = { 0, 0 };
                    return resultx;
                }
                //Data Formatting
                List<String> vaccines = InputData.GetRange(2, InputData.Count - 2);
                List<String> Prefix_ = InputData.GetRange(0, 2);
                //Get Unique Items list
                HashSet<String> uniqueItems = new HashSet<String>();
                foreach (var transaction in vaccines)
                {
                    uniqueItems.Add(transaction);
                }
                List<List<String>> subsets_total = new List<List<String>>();

                foreach (var subset in uniqueItems)
                {
                    List<String> row = new List<String>();
                    row.AddRange(Prefix_);
                    row.Add(subset);
                    subsets_total.Add(row); // Add the row to the subsets_total list
                }

                float confidence = 1;
                float MinTypedict_cluster = 6;
                foreach (var subsets in subsets_total)
                {
                    var conf = CalculateItemConfidenceWithNextItem(subsets, NextItem, rules, TypeDict);
                    if (conf[0] == 0)
                    {
                        Console.WriteLine("Can't find any record about using " + string.Join(", ", subsets) + " with " + NextItem);
                        Console.WriteLine("Warning: This item is not allowed");
                        float[] resultn = { 0, -1};
                        return resultn;
                    }
                    confidence += conf[0];
                    if(MinTypedict_cluster == 6) MinTypedict_cluster = conf[1];
                    else if ((TypeDict[MinTypedict_cluster] > TypeDict[conf[1]])) MinTypedict_cluster = conf[1]; //lấy cluster nhỏ nhất trong các cặp vaccines
                }
                Console.WriteLine("--------------");
                float[] result ={(confidence) / subsets_total.Count,TypeDict[MinTypedict_cluster] }; // nếu dương thì là có thể
                return result;
            }

            static float CalculateItemSupport(List<string> item, List<List<string>> transactions)
            {
                int count = 0;
                foreach (var transaction in transactions)
                {
                    if (IsSubset(item, transaction))
                    {
                        count++;
                    }
                }
                float support = (float)count / transactions.Count;
                return (float)Math.Round(support * 100);
            }

            static float CalculateItemConfidence(List<string> item, List<AssociationRule> rules)
            {
                float confidence = 0;

                foreach (var rule in rules)
                {
                    if (NonsequenceEqual(rule.Antecedent, item))
                    {
                        confidence = (float)Math.Round(rule.Confidence * 100);
                        break;
                    }
                }

                return confidence;
            }

            static float[] CalculateItemConfidenceWithNextItem(List<string> item, String NextItem, List<AssociationRule> rules,Dictionary<float,float> typedict)
            {
                float MaxType_cluster = -1;
                float confidence = 0;
                item.Add(NextItem);
                foreach (var rule in rules)
                {

                    List<String> ruletotal = new List<String>();
                    ruletotal.AddRange(rule.Antecedent);
                    ruletotal.Add(rule.Consequent[0]);
                    if (NonsequenceEqual(item, ruletotal))
                    {
                        Console.WriteLine(rule.ToString());
                        confidence = rule.Confidence;
                        if (MaxType_cluster == -1) MaxType_cluster = rule.Cluster;
                        else if (typedict[MaxType_cluster] < typedict[rule.Cluster]) MaxType_cluster = rule.Cluster;
                        //break;
                    }
                }
                Console.WriteLine("Max Type:"+typedict[MaxType_cluster] +" with cluster:"+ MaxType_cluster);
                float[]  result = {confidence, MaxType_cluster };
                return result;
            }

            static bool NonsequenceEqual(List<String> A, List<String> B)
            {
                HashSet<String> set1 = new HashSet<String>(A);
                HashSet<String> set2 = new HashSet<String>(B);
                return set2.All(element => set1.Contains(element));
            }

            public static List<AssociationRule> GenerateAssociationRules(List<List<string>> transactions, float minSupport, float minConfidence, int supportControl = 100)
            {
                // Tính toán support cho tất cả các mục            
                Dictionary<string, float> itemSupports = CalculateItemSupports(transactions);
                string json2 = JsonSerializer.Serialize(itemSupports);
                try
                {
                    using (StreamWriter writer = new StreamWriter("C:\\Khoa\\Git\\Association-Rule-Mining\\ItemSup.csv"))
                    {
                        writer.Write(json2);
                    }

                    Console.WriteLine("- Item support File written successfully");
                }
                catch (Exception ex)
                {
                    Console.WriteLine("- Exception message: " + ex.Message);
                }
                //tách cột ngày tuổi và giới tính ra để xử lý riêng
                List<List<string>> transactions_vc = VaccineOnly(transactions);
                // Tạo tập hợp chứa tất cả các mục duy nhất
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();
                Console.WriteLine("1.Making uniqueItems:");
                HashSet<string> uniqueItems = new HashSet<string>();
                foreach (var transaction in transactions_vc)
                {
                    foreach (var item in transaction)
                    {
                        uniqueItems.Add(item);
                    }
                }
                Console.WriteLine(">>> " + uniqueItems.Count + " Done in: " + stopwatch.Elapsed);
                // Tìm tất cả các tập con có thể
                Console.WriteLine("2.Making Subsets:");
                stopwatch = new Stopwatch();
                stopwatch.Start();
                List<List<string>> subsets_vc = GenerateSubsets(uniqueItems, 2);
                List<List<string>> subsets = GenerateSubsets_total(subsets_vc);
                subsets.RemoveAll(list => list.Count < 3);
                Console.WriteLine(">>> " + subsets.Count + " Done in: " + stopwatch.Elapsed);
                // Tính toán support cho tất cả các tập con
                Console.WriteLine("3.Calculating Subsets Support:");
                stopwatch = new Stopwatch();
                stopwatch.Start();
                Dictionary<List<string>, float> subsetSupports = CalculateSubsetSupports(subsets, transactions, supportControl); //Các tập con nếu support nhỏ hơn số lượng quy định sẽ bị loại bỏ. Support để thấp để chỉ loại bỏ trường hợp ko có trong dữ liệu
                Console.WriteLine(">>> " + " Done in: " + stopwatch.Elapsed);
                // Tạo danh sách chứa các luật kết hợp
                List<AssociationRule> rules = new List<AssociationRule>();
                Console.WriteLine("4.Finding Rules");
                stopwatch = new Stopwatch();
                stopwatch.Start();
                // Tìm các luật kết hợp
                int i = 0;
                foreach (var subset in subsets)
                {
                    if (subset.Count < 4) continue;
                    for (int t = 2; t < subset.Count; t++)
                    {
                        var item = subset[t];
                        List<string> antecedent = subset.DeepClone();
                        antecedent.RemoveAt(t);
                        List<string> consequent = new List<string> { item };

                        if (subsetSupports.ContainsKey(subset))
                        {
                            // Tính toán support và confidence cho luật kết hợp
                            float support = subsetSupports[subset];
                            float confidence = subsetSupports[subset] / itemSupports[item];
                            // Kiểm tra điều kiện minSupport và minConfidence
                            if (support >= minSupport && confidence > minConfidence)
                            {
                                AssociationRule rule = new AssociationRule(antecedent, consequent, support, confidence,(float)Math.Round(support*transactions.Count));
                                if (!rules.Contains(rule)) rules.Add(rule);
                            }
                        }
                    }
                    i++;
                }
                Console.WriteLine();
                Console.WriteLine(">>> " + rules.Count + " Done in: " + stopwatch.Elapsed);
                return rules;
            }

            public class ClusterPrediction
            {
                [ColumnName("PredictedLabel")]
                public uint PredictedClusterId;

                [ColumnName("Score")]
                public float[]? Distances;
            }

            static Dictionary<string, float> CalculateItemSupports(List<List<string>> transactions)
            {
                Dictionary<string, float> itemSupports = new Dictionary<string, float>();
                foreach (var transaction in transactions)
                {
                    foreach (var item in transaction)
                    {
                        if (itemSupports.ContainsKey(item))
                        {
                            itemSupports[item]++;
                        }
                        else
                        {
                            itemSupports[item] = 1;
                        }
                    }
                }
                int totalTransactions = transactions.Count;
                foreach (var item in itemSupports.Keys.ToList())
                {
                    float support = itemSupports[item] / totalTransactions;
                    itemSupports[item] = support;
                }
                return itemSupports;
            }

            static List<List<string>> GenerateSubsets(HashSet<string> items, int maxSubsetLength)
            {
                List<string> itemList = items.ToList();
                List<List<string>> subsets = new List<List<string>>();
                List<string> currentSubset = new List<string>();
                GenerateSubsetsRecursive(itemList, 0, currentSubset, subsets, maxSubsetLength);
                return subsets;
            }

            static List<List<string>> GenerateSubsets_total(List<List<string>> subsets)
            {
                List<List<string>> subsets_total = new List<List<string>>();
                List<List<string>> pres = new List<List<string>>();
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 6; j++)
                    {
                        List<string> item = new List<string>();
                        string group = "";
                        switch (j)
                        {
                            case 0: group = "NB"; break;
                            case 1: group = "LC"; break;
                            case 2: group = "U6"; break;
                            case 3: group = "U18"; break;
                            case 4: group = "U47"; break;
                            case 5: group = "OLD";break;
                        }
                        item.Add(i.ToString());
                        item.Add(group);
                        pres.Add(item);
                    }
                foreach (List<string> subset in subsets)
                    foreach (List<string> pre in pres)
                    {
                        if (subset.Count > 0)
                        {
                            List<string> row = new List<string>();
                            row.AddRange(pre);
                            row.AddRange(subset);
                            subsets_total.Add(row);
                        }
                    }
                Console.WriteLine(subsets.Count + "=>" + subsets_total.Count + "| Done");
                return subsets_total;
            }


            static void GenerateSubsetsRecursive(List<string> items, int index, List<string> currentSubset, List<List<string>> subsets, int maxSubsetLength = 4)
            {
                if (currentSubset.Count <= maxSubsetLength)
                {
                    subsets.Add(new List<string>(currentSubset));
                }

                if (currentSubset.Count >= maxSubsetLength)
                {
                    return;
                }

                for (int i = index; i < items.Count; i++)
                {
                    currentSubset.Add(items[i]);
                    GenerateSubsetsRecursive(items, i + 1, currentSubset, subsets, maxSubsetLength);
                    currentSubset.RemoveAt(currentSubset.Count - 1);
                }
            }


            static Dictionary<List<string>, float> CalculateSubsetSupports(List<List<string>> subsets, List<List<string>> transactions, int support_control)
            {
                Dictionary<List<string>, float> subsetSupports = new Dictionary<List<string>, float>();
                object lockObject = new object();
                Parallel.ForEach(subsets, subset =>
                {
                    if (subset != null)
                    {
                        float count = 0;
                        foreach (var transaction in transactions)
                        {
                            if (IsSubset(subset, transaction))
                            {
                                count++;
                            }
                        }
                        if (count <= support_control)
                        {
                            count = 0;
                        }
                        if (count != 0) { Console.Write("|"); }
                        else Console.Write("-");
                        int totalTransactions = transactions.Count;
                        float support = count / totalTransactions; //------------------ Suy nghĩ về việc giảm số lượng
                        lock (lockObject)
                        {
                            if (count != 0) subsetSupports.Add(subset, support);
                        }
                    }
                });
                return subsetSupports;
            }

            static bool IsSubset(List<string> subset, List<string> transaction)
            {
                if (subset == null) return false;
                foreach (var item in subset)
                {
                    if (!transaction.Contains(item))
                    {
                        return false;
                    }
                }
                return true;
            }
        }
    }
}
