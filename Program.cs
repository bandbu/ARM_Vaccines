using Accord.MachineLearning.Performance;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text.Json;
using System.Transactions;

namespace AssociationRuleMining
{
    class Program
    {
        static void Main(string[] args)
        {
            // Dữ liệu mẫu
            //List<List<string>> transactions = new List<List<string>>
            //    {
            //        new List<string>() { "vaccine 1", "vaccine 3", "vaccine 5", "vaccine 6" },
            //        new List<string>() { "vaccine 1", "vaccine 5"},
            //        new List<string>() { "vaccine 1", "vaccine 2"},
            //        new List<string>() { "vaccine 1", "vaccine 2", "vaccine 5", "vaccine 6" },
            //        new List<string>() { "vaccine 1", "vaccine 4", "vaccine 5" },
            //        new List<string>() { "vaccine 1", "vaccine 3", "vaccine 5" },
            //        new List<string>() { "vaccine 2", "vaccine 3" },
            //    };

            List<List<string>> transactions = DataReading("C:\\Users\\pc\\Downloads\\VNVCdata\\data2.csv");



            double minSupport = 0;
            double minConfidence = 0.1;
            int suportControl = 0;

            //Tính toán luật kết hợp
            Console.WriteLine("Begin Trainning");
            List<AssociationRule> rules = GenerateAssociationRules(transactions, minSupport, minConfidence, suportControl);
            Console.WriteLine("______________________________________________");
            Console.WriteLine(">> Begin Write To File");

            WriteToFile("C:\\Users\\pc\\Downloads\\VNVCdata\\Rules.json", rules);

            //In ra các luật kết hợp
            foreach (var rule in rules)
            {
                Console.WriteLine(rule.ToString());
            }

            //-------------------------------------
            List<string> inputItem = new List<string> { "vaccine 1", "vaccine 2" };
            Predict(rules, transactions, inputItem, 0.6);

        }

        //-----------------------------------
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

        static void WriteToFile(string filePath, List<AssociationRule> rules)
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

        static List<AssociationRule> ReadFromFile(string filePath)
        {
            using (StreamReader reader = new StreamReader(filePath))
            {
                string json = reader.ReadToEnd();

                List<AssociationRule> rules = JsonSerializer.Deserialize<List<AssociationRule>>(json);

                return rules;
            }
        }

        static List<List<String>> DataReading(String filePath)
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
                    for(int i=0;i<row.Count;i++)
                    {
                        string[] elements = row[i].Split('-');
                        row[i] = elements[0];
                    }
                    data.Add(row);
                }
            }

            return data;
        }

        static void Predict(List<AssociationRule> rules, List<List<string>> transactions, List<string> inputItem, double minConfidence)
        {
            double inputItemSupport = CalculateItemSupport(inputItem, transactions);
            double inputItemConfidence = CalculateItemConfidence(inputItem, rules);
            Console.WriteLine("---------------------------------");
            Console.WriteLine($"{string.Join(", ", inputItem)} ({inputItemSupport}% | {inputItemConfidence}%)");

            // Gợi ý mặt hàng tiếp theo
            Dictionary<List<string>, double> nextItems = GetNextItems(inputItem, transactions, rules, minConfidence);
            foreach (var nextItem in nextItems)
            {
                Console.WriteLine($"The next item will be: {string.Join(", ", nextItem.Key)} ({nextItem.Value}%)");
            }
        }

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
                if (rule.Antecedent.SequenceEqual(item))
                {
                    confidence = Math.Round(rule.Confidence * 100);
                    break;
                }
            }

            return confidence;
        }

        static Dictionary<List<string>, double> GetNextItems(List<string> item, List<List<string>> transactions, List<AssociationRule> rules, double minConfident)
        {
            Dictionary<List<string>, double> nextItems = new Dictionary<List<string>, double>();

            foreach (var rule in rules)
            {
                if (_IsSubset(item, rule.Antecedent) && rule.Confidence >= minConfident)
                {
                    nextItems.Add(rule.Consequent, Math.Round(rule.Confidence * 100));
                }
            }

            // Loại bỏ các mục đã có trong item
            foreach (var nextItem in nextItems)
            {
                if (_IsSubset2(item, nextItem.Key))
                {
                    nextItems.Remove(nextItem.Key);
                }
            }
            //nextItems = nextItems.Except(item).ToList();
            //nextItems = nextItems.Where(x => x.Count() <= item.Count()).ToList();

            var distinctItems = nextItems.GroupBy(x => x.Key)
                                     .Select(x => x.First())
                                     .OrderByDescending(x => x.Value)
                                     .ToList();

            return new Dictionary<List<string>, double>(distinctItems);
        }

        static bool _IsSubset(List<string> subset, List<string> superset)
        {
            int subsetIndex = 0;
            int supersetIndex = 0;

            while (subsetIndex < subset.Count && supersetIndex < superset.Count)
            {
                if (subset[subsetIndex] == superset[supersetIndex])
                {
                    subsetIndex++;
                }

                supersetIndex++;
            }

            return subsetIndex == subset.Count;
        }

        static bool _IsSubset2(List<string> subset, List<string> superset)
        {
            foreach (var subs in subset)
            {
                if (superset.Contains(subs)) return true;
            }
            return false;
        }

        static List<AssociationRule> GenerateAssociationRules(List<List<string>> transactions, double minSupport, double minConfidence, int supportControl = 100)
        {
            // Tính toán support cho tất cả các mục
            Dictionary<string, double> itemSupports = CalculateItemSupports(transactions);

            // Tạo tập hợp chứa tất cả các mục duy nhất
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            Console.WriteLine("1.Making uniqueItems:");
            HashSet<string> uniqueItems = new HashSet<string>();
            foreach (var transaction in transactions)
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
            List<List<string>> subsets = GenerateSubsets2(transactions);
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
                Console.Write("|");
                foreach (var item in subset)
                {
                    List<string> antecedent = subset;
                    List<string> consequent = new List<string> { item };

                    // Tính toán support và confidence cho luật kết hợp
                    double support = subsetSupports[subset];
                    double confidence = subsetSupports[subset] / itemSupports[item];

                    // Kiểm tra điều kiện minSupport và minConfidence
                    if (support >= minSupport && confidence >= minConfidence)
                    {
                        AssociationRule rule = new AssociationRule(antecedent, consequent, support, confidence);
                        if (!rules.Contains(rule)) rules.Add(rule);
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

        static List<List<string>> GenerateSubsets(HashSet<string> uniqueItems) //slow ver
        {
            List<List<string>> subsets = new List<List<string>>();
            subsets.Add(new List<string>());

            Parallel.ForEach(uniqueItems, item =>
            {
                int subsetCount = subsets.Count;
                for(int i=0; i<subsetCount; i++)
                {
                    if (subsets[i] != null)
                    {
                        List<string> subset = new List<string>(subsets[i]);
                        subset.Add(item);
                        subsets.Add(subset);
                    }
                };
                //Console.WriteLine(subsets.Count);
            });

            return subsets;
        }

        static List<List<string>> FindAllSubsets(HashSet<string> items) //speed up ver
        {
            List<List<string>> subsets = new List<List<string>>();

            int itemCount = items.Count;
            string[] itemsArray = new string[itemCount];
            items.CopyTo(itemsArray);

            // Tìm tất cả các tập con sử dụng parallelization
            Parallel.For(0, 1 << itemCount, i =>
            {
                List<string> subset = new List<string>();

                for (int j = 0; j < itemCount; j++)
                {
                    if ((i & (1 << j)) != 0)
                    {
                        subset.Add(itemsArray[j]);
                    }
                }

                lock (subsets)
                {
                    subsets.Add(subset);
                }
            });

            return subsets;
        }

        static Dictionary<List<string>, double> CalculateSubsetSupports(List<List<string>> subsets, List<List<string>> transactions, int support_control)
        {
            Dictionary<List<string>, double> subsetSupports = new Dictionary<List<string>, double>();
            //Parallel.ForEach(subsets, subset =>
            //{
            //    double count = 0;
            //    foreach (var transaction in transactions)
            //    {
            //        if (IsSubset(subset, transaction))
            //        {
            //            count++;
            //        }
            //    }
            //    if (count <= support_control)
            //    {
            //        subsetSupports[subset] = 0;
            //    }
            //    else
            //    {
            //        int totalTransactions = transactions.Count;
            //        double support = count / totalTransactions;
            //        subsetSupports[subset] = support;
            //    }
            //});
            foreach (var subset in subsets)
            {
                if(subset==null) continue;
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
                    subsetSupports[subset] = 0;
                    continue;
                }
                int totalTransactions = transactions.Count;
                double support = count / totalTransactions;
                subsetSupports[subset] = support;
            }
            return subsetSupports;
        }

        static bool IsSubset(List<string> subset, List<string> transaction)
        {
            if(subset==null) return false;
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

    class AssociationRule
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
}
