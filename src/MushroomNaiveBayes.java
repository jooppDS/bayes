import java.io.*;
import java.util.*;

public class MushroomNaiveBayes {

    // Store counts of classes
    private Map<String, Integer> classCounts = new HashMap<>();
    private int totalSamples = 0;

    // featureCounts.get(class).get(featureIndex).get(featureValue) = count
    private Map<String, Map<Integer, Map<String, Integer>>> featureCounts = new HashMap<>();

    // Store all distinct values per feature
    private Map<Integer, Set<String>> featureValues = new HashMap<>();

    // Train the classifier
    public void train(List<String[]> data) {
        totalSamples = data.size();

        for (String[] row : data) {
            String cls = row[0];
            classCounts.put(cls, classCounts.getOrDefault(cls, 0) + 1);

            for (int i = 1; i < row.length; i++) {
                String val = row[i];

                featureValues.putIfAbsent(i - 1, new HashSet<>());
                featureValues.get(i - 1).add(val);

                featureCounts.putIfAbsent(cls, new HashMap<>());
                featureCounts.get(cls).putIfAbsent(i - 1, new HashMap<>());
                Map<String, Integer> valCountMap = featureCounts.get(cls).get(i - 1);
                valCountMap.put(val, valCountMap.getOrDefault(val, 0) + 1);
            }
        }
    }

    // Predict class for one sample
    public String predict(String[] sample) {
        double maxLogProb = Double.NEGATIVE_INFINITY;
        String bestClass = null;

        int nClasses = classCounts.size();

        for (String cls : classCounts.keySet()) {
            // Prior with Laplace smoothing
            double prior = (classCounts.get(cls) + 1.0) / (totalSamples + nClasses);
            double logProb = Math.log(prior);

            for (int i = 1; i < sample.length; i++) {
                String val = sample[i];

                Map<Integer, Map<String, Integer>> clsFeatureCounts = featureCounts.get(cls);
                Map<String, Integer> valCountMap = clsFeatureCounts.get(i - 1);

                int valCount = valCountMap.getOrDefault(val, 0);
                int totalFeatureCount = valCountMap.values().stream().mapToInt(Integer::intValue).sum();
                int nValues = featureValues.get(i - 1).size();

                // Likelihood with Laplace smoothing
                double likelihood = (valCount + 1.0) / (totalFeatureCount + nValues);
                logProb += Math.log(likelihood);
            }

            if (logProb > maxLogProb) {
                maxLogProb = logProb;
                bestClass = cls;
            }
        }

        return bestClass;
    }

    // Evaluate on test set
    public void evaluate(List<String[]> testData) {
        int tp = 0, tn = 0, fp = 0, fn = 0;

        for (String[] row : testData) {
            String trueCls = row[0];
            String predCls = predict(row);

            if (trueCls.equals("p")) {
                if (predCls.equals("p")) tp++;
                else fn++;
            } else { // trueCls == "e"
                if (predCls.equals("e")) tn++;
                else fp++;
            }
        }

        int total = testData.size();
        double accuracy = (double)(tp + tn) / total;
        double precision = tp + fp == 0 ? 0 : (double) tp / (tp + fp);
        double recall = tp + fn == 0 ? 0 : (double) tp / (tp + fn);
        double fMeasure = (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);

        System.out.printf("Accuracy: %.4f\n", accuracy);
        System.out.printf("Precision: %.4f\n", precision);
        System.out.printf("Recall: %.4f\n", recall);
        System.out.printf("F-measure: %.4f\n", fMeasure);
    }

    // Utility to load data from a file
    public static List<String[]> loadData(String filename) throws IOException {
        List<String[]> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split(",");
                data.add(parts);
            }
        }
        return data;
    }

    public static void main(String[] args) {
        try {
            String trainFile = "agaricus-lepiota.data";
            String testFile = "agaricus-lepiota.test.data";

            List<String[]> trainData = loadData(trainFile);
            List<String[]> testData = loadData(testFile);

            MushroomNaiveBayes clf = new MushroomNaiveBayes();
            clf.train(trainData);
            clf.evaluate(testData);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
