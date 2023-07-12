using Accord.IO;
using Accord.Math;
using System.Data;
using System.Diagnostics;
using System.Text.Json;
using System.Linq;

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
            string dataPath = "D:\\Git\\Association-Rule-Mining\\data2.csv";
            string rulesPath = "D:\\Git\\Association-Rule-Mining\\Rules.json";
            List<AssociationRule> rules_= new List<AssociationRule>();
            while (inMenu)
            {
                Console.WriteLine("============== QA Co-pilot Zero ===============");
                Console.WriteLine("1. Train Co-pilot");
                Console.WriteLine("2. Load and run Train Co-pilot");
                if (rules_.Count>0)
                {
                    Console.WriteLine("3. Print all rules");
                }
                Console.WriteLine("0. Exit");
                
                Console.WriteLine("Choose an option:");
                int choice=Convert.ToInt16(Console.ReadLine());
                switch (choice)
                {
                    case 0: inMenu = false; break;
                    case 1:
                        {
                            #region Trainning Session
                            List<List<string>> transactions = QACoPilotZero.DataReading(dataPath);
                            Console.WriteLine("Data read");
                            double minSupport = 0;
                            double minConfidence = 0;
                            int suportControl = 0;

                            //Tính toán luật kết hợp
                            Console.WriteLine("Begin Trainning");
                            List<AssociationRule> rules = QACoPilotZero.GenerateAssociationRules(transactions, minSupport, minConfidence, suportControl);
                            rules_ = rules.ToList();
                            Console.WriteLine("______________________________________________");
                            Console.WriteLine(">> Begin Write To File");

                            QACoPilotZero.WriteToFile(rulesPath, rules);

                            #endregion
                        }
                        break;
                    case 2:
                        {
                            #region Using Session
                            List<AssociationRule> rules = QACoPilotZero.ReadFromFile(rulesPath);
                            rules_ = rules.ToList();
                            List<string> inputItem = new List<string> { "0", "U6", "7", "1000024" };
                            String NextItem = "4";
                            Console.WriteLine("(" + string.Join(", ", inputItem) + ") + " + NextItem + " (Confidence=" + Math.Round(QACoPilotZero.AvailableChecking(inputItem, NextItem, rules)) + "%)");
                            #endregion
                        }
                        break;
                    case 3:
                        {
                            foreach (var rule in rules_)
                            {
                                Console.WriteLine(rule.ToString());
                            }
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
        public double Support { get; }
        public double Confidence { get; }

        public AssociationRule(List<string> antecedent, List<string> consequent, double support, double confidence)
        {
            Antecedent = antecedent;
            Consequent = consequent;
            Support = support;
            Confidence = confidence;
        }

        public override string ToString()
        {
            string antecedentString = string.Join(", ", Antecedent);
            string consequentString = string.Join(", ", Consequent);

            return $"{antecedentString} => {consequentString} (support: {Support}, confidence: {Confidence})";
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
            static List<List<string>> GenerateSubsets2(List<List<string>> transactions)
            {
                HashSet<string> uniqueItems = new HashSet<string>();

                foreach (var transaction in transactions)
                {
                    foreach (var item in transaction)
                    {
                        uniqueItems.Add(item);
                    }
                }

                List<List<string>> subsets = new List<List<string>>();
                int numItems = uniqueItems.Count;

                for (int i = 1; i <= numItems; i++)
                {
                    GenerateSubsetsHelper(transactions, uniqueItems.ToList(), i, new List<string>(), subsets);
                }

                return subsets;
            }

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
                        else row[1] = "U18";
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
            public static double AvailableChecking(List<String> InputData, String NextItem, List<AssociationRule> rules)
            {
                Console.WriteLine("--------------");
                if (InputData.Count < 3) return 0;
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
                //subsets_total.RemoveAll(list => list.Count <= 3);

                double confidence = 1;

                foreach (var subsets in subsets_total)
                {
                    var conf = CalculateItemConfidenceWithNextItem(subsets, NextItem, rules);
                    if (conf == 0)
                    {
                        Console.WriteLine("Can't find any record about using " + string.Join(", ", subsets) + " with " + NextItem);
                        Console.WriteLine("Warning: This item is not allowed");
                        return 0;
                    }
                    confidence += conf;
                }
                Console.WriteLine("--------------");
                return (confidence) / subsets_total.Count; // nếu dương thì là có thể
            }

            //static String PredictNext(List<String> InputData, List<AssociationRule> rules)
            //{
            //    for
            //    return "";
            //}


            static double CalculateItemSupport(List<string> item, List<List<string>> transactions)
            {
                int count = 0;

                foreach (var transaction in transactions)
                {
                    if (IsSubset(item, transaction))
                    {
                        count++;
                    }
                }

                double support = (double)count / transactions.Count;
                return Math.Round(support * 100);
            }

            static double CalculateItemConfidence(List<string> item, List<AssociationRule> rules)
            {
                double confidence = 0;

                foreach (var rule in rules)
                {
                    if (NonsequenceEqual(rule.Antecedent, item))
                    {
                        confidence = Math.Round(rule.Confidence * 100);
                        break;
                    }
                }

                return confidence;
            }

            static double CalculateItemConfidenceWithNextItem(List<string> item, String NextItem, List<AssociationRule> rules)
            {
                double confidence = 0;
                List<String> nextItems = new List<String>();
                nextItems.Add(NextItem);
                foreach (var rule in rules)
                {
                    if (NonsequenceEqual(rule.Antecedent, item) && NonsequenceEqual(rule.Consequent, nextItems))
                    {
                        Console.WriteLine(rule.ToString());
                        confidence = Math.Round(rule.Confidence * 100);
                        break;
                    }
                }
                return confidence;
            }

            static bool NonsequenceEqual(List<String> A, List<String> B)
            {
                HashSet<String> set1 = new HashSet<String>(A);
                HashSet<String> set2 = new HashSet<String>(B);
                return set2.All(element => set1.Contains(element));
            }

            public static List<AssociationRule> GenerateAssociationRules(List<List<string>> transactions, double minSupport, double minConfidence, int supportControl = 100)
            {
                // Tính toán support cho tất cả các mục            
                Dictionary<string, double> itemSupports = CalculateItemSupports(transactions);
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
                Dictionary<List<string>, double> subsetSupports = CalculateSubsetSupports(subsets, transactions, supportControl); //Các tập con nếu support nhỏ hơn số lượng quy định sẽ bị loại bỏ. Support để thấp để chỉ loại bỏ trường hợp ko có trong dữ liệu
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
                            double support = subsetSupports[subset];
                            double confidence = subsetSupports[subset] / itemSupports[item];
                            // Kiểm tra điều kiện minSupport và minConfidence
                            if (support >= minSupport && confidence > minConfidence)
                            {
                                AssociationRule rule = new AssociationRule(antecedent, consequent, support, confidence);
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

            static Dictionary<string, double> CalculateItemSupports(List<List<string>> transactions)
            {
                Dictionary<string, double> itemSupports = new Dictionary<string, double>();

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
                    double support = itemSupports[item] / totalTransactions;
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
                    for (int j = 0; j < 4; j++)
                    {
                        List<string> item = new List<string>();
                        string group = "";
                        switch (j)
                        {
                            case 0: group = "NB"; break;
                            case 1: group = "LC"; break;
                            case 2: group = "U6"; break;
                            case 3: group = "U18"; break;
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


            static Dictionary<List<string>, double> CalculateSubsetSupports(List<List<string>> subsets, List<List<string>> transactions, int support_control)
            {
                Dictionary<List<string>, double> subsetSupports = new Dictionary<List<string>, double>();
                object lockObject = new object();
                Parallel.ForEach(subsets, subset =>
                {
                    if (subset != null)
                    {
                        double count = 0;
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
                        if(count!=0) { Console.Write("|"); }
                        else Console.Write("-");
                        int totalTransactions = transactions.Count;
                        double support = count / totalTransactions;
                        lock (lockObject)
                        {
                            if(count!=0) subsetSupports.Add(subset, support);
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
