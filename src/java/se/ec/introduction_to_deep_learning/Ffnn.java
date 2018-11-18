package se.ec.introduction_to_deep_learning;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;

import java.io.IOException;
import java.util.Arrays;

public class Ffnn {

    private static Logger LOGGER = LoggerFactory.getLogger(Ffnn.class);

    public static void main(String[] args) throws IOException, InterruptedException {
        double trainTestSplit = 0.5;
        int hiddenLayerSize = 30;


        final int numLinesToSkip = 1;
        final char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("data.csv").getFile()));

        // Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        final int labelIndex = 3;
        final int numClasses = 2;
        final int batchSize = 150;

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(trainTestSplit);

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

//        DataSet trainingData = DataSet.merge(allData.asList().subList(0, 7));
//        DataSet testData = DataSet.merge(allData.asList().subList(7, allData.asList().size()));

        // We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

        final int numInputs = 3;
        final long seed = 6;
        final int outputNum = 2;

        LOGGER.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(1e-4)
                .activation(Activation.SIGMOID)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(hiddenLayerSize)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .nIn(hiddenLayerSize).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build();

        // run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        for (int i = 0; i < 1500; i++) {
            model.fit(trainingData);
        }

        // evaluate the model on the test set
        Evaluation eval = new Evaluation(outputNum);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        LOGGER.info(eval.stats());


        // make predictions
        RecordReader predictRecordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        predictRecordReader.initialize(new FileSplit(new ClassPathResource("new_data.csv").getFile()));
        DataSetIterator predictIterator = new RecordReaderDataSetIterator(predictRecordReader, batchSize);
        DataSet predictData = predictIterator.next();
        normalizer.transform(predictData);

        int[] predict = model.predict(predictData.getFeatures());
        LOGGER.info("Predict: " + Arrays.toString(predict));
    }


}
