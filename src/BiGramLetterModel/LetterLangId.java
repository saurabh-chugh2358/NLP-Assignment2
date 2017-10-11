package BiGramLetterModel;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;


public class LetterLangId {
	private ArrayList<Hashtable<String,Integer>>frequencyMatrix=new ArrayList<Hashtable<String,Integer>>();
	private int[] totalTokens;
	private int nGramLen;
	private String lm="noSmoothing";

	public static void main(String[] args) throws IOException {
		int nGramLength=2;
		String[] trainingFiles = new String[]{"HW2-english.txt", "HW2-french.txt", "HW2-german.txt"};
		String[] langArray = new String[]{"EN", "FR", "GR"};		
		Hashtable<String,Integer>languageIndx=new Hashtable<String,Integer>();
		languageIndx.put("EN", 0);
		languageIndx.put("FR", 1);
		languageIndx.put("GR", 2);
		String[] outputLMs = new String[]{"EN_Bi_Letter.bin", "FR_Bi_Letter.bin", "GR_Bi_Letter.bin"};


		LetterLangId[] langModels=new LetterLangId[trainingFiles.length];
		for(int i=0;i<langModels.length;i++){
			langModels[i]=new LetterLangId(trainingFiles[i], outputLMs[i],nGramLength);
		}

		BufferedReader readbuffer = new BufferedReader(new InputStreamReader(new FileInputStream("LangID.gold.txt")));
		String wrkStr=readbuffer.readLine();//read and skip the header line in the gold-ID file
		ArrayList<String>goldLangIDs=new ArrayList<String>();
		wrkStr=readbuffer.readLine();
		while (wrkStr!=null){	
			wrkStr=wrkStr.trim();
			if(!wrkStr.isEmpty()){//to skip empty lines
				goldLangIDs.add(wrkStr.split("\\s+")[1]);
				wrkStr=readbuffer.readLine();
			}
		}	
		readbuffer.close();

		readbuffer = new BufferedReader(new InputStreamReader(new FileInputStream("LangID.test.txt")));
		ArrayList<String>tstString=new ArrayList<String>();
		wrkStr=readbuffer.readLine();
		while (wrkStr!=null){
			wrkStr=wrkStr.trim();
			if(!wrkStr.isEmpty()){//to skip empty lines
				tstString.add(wrkStr.replaceAll("^\\d+.\\s", ""));//remove the line index before adding the line into the test sentences array
				wrkStr=readbuffer.readLine();
			}
		}	
		readbuffer.close();

		BufferedWriter finalOuput = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("BigramLetterLangId.out"), "UTF-8"));
		finalOuput.write("Lang-Id\n");//write the header
		double finalScore=Double.NEGATIVE_INFINITY;
		String calcId=langArray[0];
		int accurate=0;
		int[][]confusionMatrix = new int[langArray.length][langArray.length];

		for(int i=0;i<tstString.size();i++){
			finalScore=langModels[0].getSentenceProbability(tstString.get(i));
			calcId=langArray[0];
			for(int j=1;j<langModels.length;j++){
				double curr=langModels[j].getSentenceProbability(tstString.get(i));
				if(finalScore<curr){
					finalScore=curr;
					calcId=langArray[j];
				}				
			}
			finalOuput.write(i+". "+calcId+"\n");
			if(calcId.equals(goldLangIDs.get(i))){
				accurate += 1;
			}else{
				confusionMatrix[languageIndx.get(goldLangIDs.get(i))][languageIndx.get(calcId)]++;
			}
		}
		double accuracy=(double)accurate/(double)tstString.size();
		System.out.println("Accuracy= "+Double.toString(accuracy));
		System.out.println("Confusion Matrix:\n\tEN\tFR\tGR");
		for(int i=0;i<langArray.length;i++){
			System.out.print(langArray[i]+"\t");
			for(int j=0;j<langArray.length;j++){
				System.out.print(confusionMatrix[i][j]+"\t");
			}
			System.out.print("\n");
		}
		finalOuput.write("\nAccuracy= "+Double.toString(accuracy));		
		finalOuput.close();
	}



	//n-grams with no smoothing
	public LetterLangId(String trainFile, String outputBinaryLmFile, int nGramLen) throws IOException {
		this.nGramLen=nGramLen;
		totalTokens=new int[nGramLen];
		for(int i=0;i<nGramLen;i++){
			frequencyMatrix.add(new Hashtable<String,Integer>());
		}

		//Building the LM
		BufferedReader readbuffer = new BufferedReader(new InputStreamReader(new FileInputStream(trainFile)));
		String line;
		int allWords=0;
		while ((line=readbuffer.readLine())!=null){
			line=preprocessingStep(line);
			if(!line.isEmpty()){
				ArrayList<String> letters = new ArrayList<String>(Arrays.asList(line.replaceAll("", "\t").split("\t")));
				letters.removeAll(Arrays.asList("", null));
				allWords+=letters.size();


				for(int i=0;i<=letters.size()-nGramLen;i++){
					int startIndx=i;
					int endIndx=startIndx+nGramLen-1;
					StringBuilder runningStr=new StringBuilder();
					for(int j=startIndx;j<=endIndx;j++){
						runningStr.append(letters.get(j).replaceAll("\\s", "@@space@@")+" ");
						int ngSize=runningStr.toString().trim().split(" ").length-1;
						Hashtable<String,Integer> tmpTable=frequencyMatrix.get(ngSize);
						int count;
						if(tmpTable.containsKey(runningStr.toString().trim())){
							count=tmpTable.get(runningStr.toString().trim()).intValue()+1;
						}else{
							count=1;
						}
						tmpTable.put(runningStr.toString().trim(),new Integer(count));
						frequencyMatrix.set(ngSize,tmpTable);
					}
				}
			}
		}
		readbuffer.close();
		totalTokens[0]=allWords;
		for(int i=1;i<totalTokens.length;i++){
			totalTokens[i]=totalTokens[i-1]-1;
		}
		finalOutputLM(outputBinaryLmFile);
	}

	private String preprocessingStep(String str){
		return str.replaceAll("([\"\'\\}\\{\\«\\»\\-\\=\\_\\:\\#\\@\\!\\?\\^\\/\\(\\)\\[\\]\\%\\;\\\\\\+\\.\\,]+)", " $1 ")
		.replaceAll("(\\d+)", " $1 ")
		.replaceAll("\\s+", " ")
		.trim()
		.toLowerCase();
	}

	public double getSentenceProbability(String str){
		str=preprocessingStep(str);
		if(str.isEmpty()){
			return Double.NEGATIVE_INFINITY;
		}
		ArrayList<String> letters = new ArrayList<String>(Arrays.asList(str.replaceAll("", "\t").split("\t")));
		letters.removeAll(Arrays.asList("", null));
		double probabiliy=Math.log10(nGramProbability(letters.get(0)));
		for(int i=0;i<=letters.size()-nGramLen;i++){
			int startIdx=i;
			int endIdx=startIdx+nGramLen-1;
			StringBuilder currStr=new StringBuilder();
			for(int j=startIdx;j<=endIdx;j++){
				currStr.append(letters.get(j).replaceAll("\\s", "@@space@@")+" ");
			}
			double currProbability=nGramProbability(currStr.toString().trim());
			if(currProbability==0){
				probabiliy+=Double.NEGATIVE_INFINITY;
			}else{
				probabiliy+=Math.log10(currProbability);
			}
		}
		return probabiliy;
	}


	private double nGramProbability(String str){
		String[]tokens=str.split("\\s+");
		int nGramSize=tokens.length;
		double matchingCount=0;
		double totalCount=0;

		if(frequencyMatrix.get(nGramSize-1).containsKey(str)){
			matchingCount=frequencyMatrix.get(nGramSize-1).get(str);
		}
		if(nGramSize>1){
			StringBuilder denominatorStr= new StringBuilder();
			for(int i=0; i<nGramSize-1;i++){
				denominatorStr.append(tokens[i]+" ");
			}
			if(frequencyMatrix.get(nGramSize-2).containsKey(denominatorStr.toString().trim())){
				totalCount=frequencyMatrix.get(nGramSize-2).get(denominatorStr.toString().trim());
			}
		}else{
			totalCount=totalTokens[0];
		}
		if(totalCount==0){
			return 0;
		}else{
			return (matchingCount/totalCount);
		}		
	}

	private void finalOutputLM(String outfile) throws IOException{
		ObjectOutput output = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(outfile)));
		output.writeObject(lm);
		output.writeObject(nGramLen);
		output.writeObject(frequencyMatrix);
		output.writeObject(totalTokens);    	
		output.close();		
	}
}
