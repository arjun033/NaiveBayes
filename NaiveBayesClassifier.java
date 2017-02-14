/**
 * THIS PROGRAM ASSUMES THAT ALL THE DATA FILES HAVE BEEN PLACED
 * UNDER THE ROOT DIRECTORY OF THE PROJECT
 * @author Arjun Bhattacharya
 */


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedHashSet;
import java.util.StringTokenizer;
import java.util.TreeMap;

public class NaiveBayesClassifier {
	
	private static int vocabWordCount = 0;
	private static int labelCount = 0;
	private static int[] n_arr;
	private static int[][] wordFreqForLabel;
	private static int[][] confusionMatrix;
	private static double[] classPrior;
	private static double[][] p_mle_arr;
	private static double[][] p_be_arr;
	private static TreeMap<Integer, Integer> docToLabelMap;
	private static TreeMap<Integer, LinkedHashSet<Integer>> labelToDocSetMap;
	private static TreeMap<Integer, LinkedHashSet<Integer>> wordsInDoc;
	
	public static void main(String[] args) {
		if (args.length != 6) {
			System.out.println("Command line input format incorrect");
			System.exit(0);
		}
		
		long startTime = System.currentTimeMillis();

		String vocabFile = args[0];
		String mapFile = args[1];
		String trainLabelFile = args[2];
		String trainDataFile = args[3];
		String testLabelFile = args[4];
		String testDataFile = args[5];
		
		//Counting total number of words in the vocabulary
		setVocabWordCount(vocabFile);
		
		//Calculating the total number of distinct classes
		setLabelCount(mapFile);
		
		//Setting the document to class map and class to document set map for training data
		setDocToLabelMap(trainLabelFile);
		
		//Setting the word frequencies for all documents for each class from training data
		setWordFreqForLabel(trainDataFile, true);
		
		//Printing class priors
		printClassPrior();
		
		//Calculating the total number of words in all documents for each class and setting it in an array
		setNumWordsInClass();
		
		//Calculating the Maximum Likelihood Estimator
		calculateMaxLikelihoodEstimator();
		
		//Calculating the Bayesian Estimator
		calculateBayesianEstimator();
		
		//Applying the Bayesian Estimator on the training data
		TreeMap<Integer, Integer> resultMapTrain = classifyDocuments(true);
		
		//Calculating and printing the overall accuracy for training data
		System.out.println("###################################### OVERALL ACCURACY (TRAINING) #######################################");
		System.out.println("##########################################################################################################");
		System.out.println("##");
		System.out.println("##  Overall Accuracy (Training): "+calculateOverallAccuracy(docToLabelMap, resultMapTrain)+"%");
		System.out.println("##");
		System.out.println("##########################################################################################################");
		
		//Calculating and printing class accuracy for training data
		TreeMap<Integer, LinkedHashSet<Integer>> resultLabelToDocSetMap = new TreeMap<>();
		for(int docId : resultMapTrain.keySet()){
			int labelId = resultMapTrain.get(docId);
			if(resultLabelToDocSetMap.containsKey(labelId)){
				resultLabelToDocSetMap.get(labelId).add(docId);
			} else {
				LinkedHashSet<Integer> newDocSet = new LinkedHashSet<>();
				newDocSet.add(docId);
				resultLabelToDocSetMap.put(labelId, newDocSet);
			}
		}
		double[] classAccuArr = calculateClassAccuracy(resultLabelToDocSetMap,labelToDocSetMap);
		System.out.println("##################################### CLASS ACCURACY (TRAINING) ##########################################");
		System.out.println("##########################################################################################################");
		System.out.println("##");
		for(int labelId=1; labelId<=classAccuArr.length; labelId++){
			System.out.println("##  Group "+labelId+": "+classAccuArr[labelId-1]);
		}
		System.out.println("##");
		System.out.println("##########################################################################################################");
		
		//Generating and printing the confusion matrix based on the training data
		generateConfusionMatrix(resultLabelToDocSetMap,labelToDocSetMap);
		printConfusionMatrix("TRAINING");
		
		//Setting the document to class map and class to document set map for test data
		setDocToLabelMap(testLabelFile);
		
		//Setting the document to word set map from test data
		setWordFreqForLabel(testDataFile, false);
		
		//Applying the Bayesian Estimator on the test data
		TreeMap<Integer, Integer> resultMapTestBE = classifyDocuments(true);
		
		//Calculating and printing the overall accuracy for test data (BE)
		System.out.println("###################################### OVERALL ACCURACY (TEST BE) ########################################");
		System.out.println("##########################################################################################################");
		System.out.println("##");
		System.out.println("##  Overall Accuracy (Test BE): "+calculateOverallAccuracy(docToLabelMap, resultMapTestBE)+"%");
		System.out.println("##");
		System.out.println("##########################################################################################################");
		
		//Calculating and printing class accuracy for test data (BE)
		resultLabelToDocSetMap = new TreeMap<>();
		for(int docId : resultMapTestBE.keySet()){
			int labelId = resultMapTestBE.get(docId);
			if(resultLabelToDocSetMap.containsKey(labelId)){
				resultLabelToDocSetMap.get(labelId).add(docId);
			} else {
				LinkedHashSet<Integer> newDocSet = new LinkedHashSet<>();
				newDocSet.add(docId);
				resultLabelToDocSetMap.put(labelId, newDocSet);
			}
		}
		classAccuArr = calculateClassAccuracy(resultLabelToDocSetMap,labelToDocSetMap);
		System.out.println("###################################### CLASS ACCURACY (TEST BE) ##########################################");
		System.out.println("##########################################################################################################");
		System.out.println("##");
		for(int labelId=1; labelId<=classAccuArr.length; labelId++){
			System.out.println("##  Group "+labelId+": "+classAccuArr[labelId-1]);
		}
		System.out.println("##");
		System.out.println("##########################################################################################################");
		
		//Generating and printing the confusion matrix based on the test data (BE)
		generateConfusionMatrix(resultLabelToDocSetMap,labelToDocSetMap);
		printConfusionMatrix(" TEST BE");
		
		//Applying the Maximum Likelihood Estimator on the test data
		TreeMap<Integer, Integer> resultMapTestMLE = classifyDocuments(false);
		
		//Calculating and printing the overall accuracy for test data (MLE)
		System.out.println("###################################### OVERALL ACCURACY (TEST MLE) #######################################");
		System.out.println("##########################################################################################################");
		System.out.println("##");
		System.out.println("##  Overall Accuracy (Test MLE): "+calculateOverallAccuracy(docToLabelMap, resultMapTestMLE)+"%");
		System.out.println("##");
		System.out.println("##########################################################################################################");
		
		//Calculating and printing class accuracy for test data (MLE)
		resultLabelToDocSetMap = new TreeMap<>();
		for(int docId : resultMapTestMLE.keySet()){
			int labelId = resultMapTestMLE.get(docId);
			if(resultLabelToDocSetMap.containsKey(labelId)){
				resultLabelToDocSetMap.get(labelId).add(docId);
			} else {
				LinkedHashSet<Integer> newDocSet = new LinkedHashSet<>();
				newDocSet.add(docId);
				resultLabelToDocSetMap.put(labelId, newDocSet);
			}
		}
		classAccuArr = calculateClassAccuracy(resultLabelToDocSetMap,labelToDocSetMap);
		System.out.println("###################################### CLASS ACCURACY (TEST MLE) #########################################");
		System.out.println("##########################################################################################################");
		System.out.println("##");
		for(int labelId=1; labelId<=classAccuArr.length; labelId++){
			System.out.println("##  Group "+labelId+": "+classAccuArr[labelId-1]);
		}
		System.out.println("##");
		System.out.println("##########################################################################################################");
		
		//Generating and printing the confusion matrix based on the test data (MLE)
		generateConfusionMatrix(resultLabelToDocSetMap,labelToDocSetMap);
		printConfusionMatrix("TEST MLE");
		
		System.out.println("############################################ EXECUTION TIME ##############################################");
		System.out.println("##########################################################################################################");
		System.out.println("##");
		long stopTime = System.currentTimeMillis();
	    double elapsedTime = (stopTime - startTime)/1000.0;
	    System.out.println("##  "+elapsedTime+"s");
	    System.out.println("##");
		System.out.println("##########################################################################################################");
	}
	
	/**
	 * This method reads the label file and produces a mapping between
	 * a document and a class. It also creates a mapping between a class
	 * and a set of documents corresponding to the class.
	 * @param fileName
	 */
	public static void setDocToLabelMap (String fileName) {
		String labelIdStr = "";
		int docId = 1;
		docToLabelMap = new TreeMap<> ();
		labelToDocSetMap = new TreeMap<>();
		try {
			BufferedReader brTrainLabelFile = new BufferedReader(new FileReader(fileName));
			while((labelIdStr = brTrainLabelFile.readLine()) != null){
				docToLabelMap.put(docId, Integer.parseInt(labelIdStr));
				if(labelToDocSetMap.containsKey(Integer.parseInt(labelIdStr))){
					labelToDocSetMap.get(Integer.parseInt(labelIdStr)).add(docId);
				} else {
					LinkedHashSet<Integer> newDocSet = new LinkedHashSet<>();
					newDocSet.add(docId);
					labelToDocSetMap.put(Integer.parseInt(labelIdStr), newDocSet);
				}
				docId++;
			}
			brTrainLabelFile.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * This method reads the data file data file and sets a 2D array
	 * where value(i,j) is the frequency of word i in class j i.e. n_k. It also produces
	 * the mapping between a document and the set of words occurring in that document.
	 * @param fileName
	 * @param train
	 */
	public static void setWordFreqForLabel (String fileName, boolean train) {
		int labelIdx = 0;
		String trainDocTuple = "";
		StringTokenizer st = null;
		wordsInDoc = new TreeMap<>();
		if(train){
			wordFreqForLabel = new int[vocabWordCount][labelCount];
		}
		try {
			int docIndex=0, wordIndex=0, wordCount=0;
			BufferedReader brTrainDataFile = new BufferedReader(new FileReader(fileName));
			while((trainDocTuple = brTrainDataFile.readLine()) != null){
				int tokenNum = 1;
				st = new StringTokenizer(trainDocTuple, ",");
				while(st.hasMoreTokens()){
					if (tokenNum==1) {
						docIndex = Integer.parseInt(st.nextToken());
					} else if (tokenNum==2) {
						wordIndex = Integer.parseInt(st.nextToken());
					} else if (tokenNum==3) {
						wordCount = Integer.parseInt(st.nextToken());
					}
					tokenNum++;
				}
				
				//Creating a mapping between documents and the set of words occurring in them
				if(wordsInDoc.containsKey(docIndex)){
					wordsInDoc.get(docIndex).add(wordIndex);
				} else {
					LinkedHashSet<Integer> newWordSet = new LinkedHashSet<>();
					newWordSet.add(wordIndex);
					wordsInDoc.put(docIndex, newWordSet);
				}
				
				//Updating the word frequency corresponding to a class in the 2D array
				if(train){
					labelIdx = docToLabelMap.get(docIndex);
					wordFreqForLabel[wordIndex-1][labelIdx-1] += wordCount;
				}
			}
			brTrainDataFile.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * This method reads the vocabulary.txt file and sets the total word count
	 * @param fileName
	 */
	public static void setVocabWordCount (String fileName) {
		String vocabWord = "";
		try {
			BufferedReader brVocabFile = new BufferedReader(new FileReader(fileName));
			while((vocabWord = brVocabFile.readLine()) != null){
				vocabWordCount++;
			}
			brVocabFile.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * This method reads the map.csv file and sets the total label count
	 * @param fileName
	 */
	public static void setLabelCount (String fileName) {
		String labelName = "";
		try {
			BufferedReader brTrainLabelFile = new BufferedReader(new FileReader(fileName));
			while((labelName = brTrainLabelFile.readLine()) != null){
				labelCount++;
			}
			brTrainLabelFile.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * This method calculates the prior probability of a document
	 * belonging to a particular class
	 * @param labelId
	 */
	public static void printClassPrior () {
		double prior = 0.0;
		classPrior = new double[labelCount];
		System.out.println("##########################################################################################################");
		System.out.println("############################################ CLASS PRIORS ################################################");
		System.out.println("##########################################################################################################");
		System.out.println("##");
		for(int labelId=1; labelId<=labelCount; labelId++){
			prior = (double)labelToDocSetMap.get(labelId).size()/(double)docToLabelMap.size();
			prior = (double)Math.round(prior * 10000d)/10000d;
			classPrior[labelId-1] = prior;
			System.out.println("##  P(Omega = "+ labelId + ") = "+ prior);
		}
		System.out.println("##");
		System.out.println("##########################################################################################################");
	}
	
	/**
	 * This method calculates the total number of words in all 
	 * documents in a class. This is simply the sum of all
	 * elements in a column of the word frequency array. The method
	 * saves the sum of frequencies in an array where the ith element
	 * corresponds to the total number of words in all documents in
	 * the (i+1)th class.
	 */
	public static void setNumWordsInClass () {
		n_arr = new int[labelCount];
		for(int i=0; i<labelCount; i++) {
			for(int j=0; j<vocabWordCount; j++){
				n_arr[i] += wordFreqForLabel[j][i];
			}
		}
	}
	
	/**
	 * This method calculates the maximum likelihood estimator and sets
	 * its value in a 2D array
	 */
	public static void calculateMaxLikelihoodEstimator () {
		p_mle_arr = new double[vocabWordCount][labelCount];
		for(int i=0; i<vocabWordCount; i++){
			for(int j=0; j<labelCount; j++){
				p_mle_arr[i][j] = (double)wordFreqForLabel[i][j]/(double)n_arr[j];
			}
		}
	}
	
	/**
	 * This method calculates the Bayesian estimator and sets its value in a 2D array
	 */
	public static void calculateBayesianEstimator () {
		p_be_arr = new double[vocabWordCount][labelCount];
		for(int i=0; i<vocabWordCount; i++){
			for(int j=0; j<labelCount; j++){
				p_be_arr[i][j] = (double)(wordFreqForLabel[i][j]+1)/(double)(n_arr[j]+vocabWordCount);
			}
		}
	}
	
	/**
	 * This method classifies the documents using the Bayesian estimator or the
	 * Maximum Likelihood Estimator based on the boolean argument value and returns
	 * a mapping of documents to classes 
	 * @param isBayesian
	 * @return
	 */
	public static TreeMap<Integer, Integer> classifyDocuments (boolean isBayesian) {
		TreeMap<Integer, Integer> classificationResult = new TreeMap<>();
		
		for(int docIdx : docToLabelMap.keySet()){
			double runningMax = (-1)*Double.MAX_VALUE;
			int classNB = 0;
			for(int labelId=1; labelId<=labelCount; labelId++) {
				double omegaNB;
				omegaNB = isBayesian ? 0.0 : 1;
				for(int wordIdx : wordsInDoc.get(docIdx)){
					if(isBayesian){
						omegaNB += Math.log(p_be_arr[wordIdx-1][labelId-1]);
					} else {
						omegaNB *= p_mle_arr[wordIdx-1][labelId-1];
					}
				}
				if(isBayesian){
					omegaNB += Math.log(classPrior[labelId-1]);
				} else {
					omegaNB *= classPrior[labelId-1];
				}
				if (omegaNB>runningMax) {
					runningMax=omegaNB;
					classNB=labelId;
				}
			}
			classificationResult.put(docIdx, classNB);
		}
		return classificationResult;
	}
	
	/**
	 * This method compares the original document to class mapping with the 
	 * generated classification and returns the accuracy of the classifier
	 * @param originalClassification
	 * @param resultClassification
	 * @return
	 */
	public static double calculateOverallAccuracy (TreeMap<Integer, Integer> originalClassification, 
			TreeMap<Integer, Integer> resultClassification) {
		double overallAccuracy = 0.0;
		int correctlyClassified = 0;
		for(int docId : originalClassification.keySet()){
			if(resultClassification.containsKey(docId) && originalClassification.get(docId)==resultClassification.get(docId)){
				correctlyClassified++;
			}
		}
		overallAccuracy = (double)(correctlyClassified*100)/(double)originalClassification.size();
		return (double)Math.round(overallAccuracy * 100d)/100d;
	}
	
	/**
	 * This method calculates the class specific accuracy and returns an array
	 * of accuracy values where the ith element is the accuracy of the (i+1)th class
	 * @param classificationResult
	 * @param originalLabelToDocSetMap
	 * @return
	 */
	public static double[] calculateClassAccuracy (TreeMap<Integer, LinkedHashSet<Integer>> resultLabelToDocSetMap, 
			TreeMap<Integer, LinkedHashSet<Integer>> originalLabelToDocSetMap){
		int correctlyClassified = 0;
		double classAccuracy = 0.0;
		double classAccuracyArr[] = new double[labelCount];
		for(int labelId : originalLabelToDocSetMap.keySet()){
			if(resultLabelToDocSetMap.containsKey(labelId)){
				LinkedHashSet<Integer> intersection = new LinkedHashSet<>(resultLabelToDocSetMap.get(labelId));
				intersection.retainAll(originalLabelToDocSetMap.get(labelId));
				correctlyClassified = intersection.size();
			}
			classAccuracy = (double)(correctlyClassified*100)/(double)originalLabelToDocSetMap.get(labelId).size();
			classAccuracyArr[labelId-1] = (double)Math.round(classAccuracy * 100d)/100d;
		}
		return classAccuracyArr;
	}
	
	/**
	 * This method calculates the confusion matrix where a cell (i,j) 
	 * in the matrix represents the number of documents in group i that 
	 * are predicted to be in group j.
	 * @param resultLabelToDocSetMap
	 * @param originalLabelToDocSetMap
	 */
	public static void generateConfusionMatrix (TreeMap<Integer, LinkedHashSet<Integer>> resultLabelToDocSetMap, 
			TreeMap<Integer, LinkedHashSet<Integer>> originalLabelToDocSetMap){
		confusionMatrix = new int[labelCount][labelCount];
		for(int origLabelId : originalLabelToDocSetMap.keySet()){
			for(int resultLabelId : resultLabelToDocSetMap.keySet()){
				LinkedHashSet<Integer> intersection = new LinkedHashSet<>(resultLabelToDocSetMap.get(resultLabelId));
				intersection.retainAll(originalLabelToDocSetMap.get(origLabelId));
				confusionMatrix[origLabelId-1][resultLabelId-1] = intersection.size();
			}
		}
	}
	
	/**
	 * This method prints the confusion matrix on the console
	 * @param s
	 */
	public static void printConfusionMatrix (String s){
		System.out.println("###################################### CONFUSION MATRIX ("+s+") #######################################");
		System.out.println("##########################################################################################################");
		System.out.println("##");
		for(int i=0; i<confusionMatrix.length; i++){
			System.out.format("%-2s", "##");
			for(int j=0; j<confusionMatrix[i].length; j++){
				System.out.format("%5d", confusionMatrix[i][j]);
			}
			System.out.println();
		}
		System.out.println("##");
		System.out.println("##########################################################################################################");
	}
}
