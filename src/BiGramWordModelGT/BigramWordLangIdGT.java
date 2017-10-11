package BiGramWordModelGT;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;


public class BigramWordLangIdGT {
	private int[] totalTokens;
	private double[] n0;
	private int maxSmoohing=5;
	private int nGramLen;
	private String lm="GoodTuringSmoohing";
	private ArrayList<Hashtable<Integer,Integer>>histogram=new ArrayList<Hashtable<Integer,Integer>>();
	private ArrayList<Hashtable<String,Integer>>frequencyMatrix=new ArrayList<Hashtable<String,Integer>>();

	public static void main(String[] args) throws IOException {
		String[] trainingFiles = new String[]{"HW2-english.txt", "HW2-french.txt", "HW2-german.txt"};
		String[] langArray = new String[]{"EN", "FR", "GR"};
		Hashtable<String,Integer>lang2Idx=new Hashtable<String,Integer>();
		lang2Idx.put("EN", 0);
		lang2Idx.put("FR", 1);
		lang2Idx.put("GR", 2);
		String[] outLMs = new String[]{"EN_Bi_Words_GT.bin", "FR_Bi_Words_GT.bin", "GR_Bi_Words_GT.bin"};
		int maxNgramSize=2;

		//Building the LMs
		BigramWordLangIdGT[] langModels=new BigramWordLangIdGT[trainingFiles.length];
		for(int i=0;i<langModels.length;i++){
			langModels[i]=new BigramWordLangIdGT(trainingFiles[i], outLMs[i],maxNgramSize);
		}

		BufferedReader readbuffer = new BufferedReader(new InputStreamReader(new FileInputStream("LangID.gold.txt")));
		String wrkStr=readbuffer.readLine();
		ArrayList<String>goldLangIDs=new ArrayList<String>();
		wrkStr=readbuffer.readLine();
		while (wrkStr!=null){	
			wrkStr=wrkStr.trim();
			if(!wrkStr.isEmpty()){
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
			if(!wrkStr.isEmpty()){
				tstString.add(wrkStr.replaceAll("^\\d+.\\s", ""));
				wrkStr=readbuffer.readLine();
			}
		}	
		readbuffer.close();

		BufferedWriter finalOuput = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("BigramLetterLangId-GT.out"), "UTF-8"));
		finalOuput.write("ID LANG\n");
		double finalScore=0;
		String calcId=langArray[0];
		int accurate=0;
		int[][]confusionMatrix = new int[langArray.length][langArray.length];

		for(int i=0;i<tstString.size();i++){
			finalScore=langModels[0].getSentenceProbability(tstString.get(i));
			calcId=langArray[0];
			for(int j=1;j<langModels.length;j++){
				double tmpScore=langModels[j].getSentenceProbability(tstString.get(i));
				if(tmpScore>finalScore){
					finalScore=tmpScore;
					calcId=langArray[j];
				}
			}
			finalOuput.write(i+". "+calcId+"\n");
			if(calcId.equals(goldLangIDs.get(i))){
				accurate += 1;
			}else{
				confusionMatrix[lang2Idx.get(goldLangIDs.get(i))][lang2Idx.get(calcId)]++;
			}
		}

		double accuracy=(double)accurate/(double)tstString.size();
		finalOuput.write("\nAccuracy= "+Double.toString(accuracy));
		finalOuput.close();

		System.out.println("Confusion Matrix:\n\tEN\tFR\tGR");
		for(int i=0;i<langArray.length;i++){
			System.out.print(langArray[i]+"\t");
			for(int j=0;j<langArray.length;j++){
				System.out.print(confusionMatrix[i][j]+"\t");
			}
			System.out.print("\n");
		}
		System.out.println("Accuracy= "+Double.toString(accuracy));
	}

	public BigramWordLangIdGT(String trainFile, String outputBinaryLmFile, int nGramLen) throws IOException{
		this.nGramLen=nGramLen;
		totalTokens=new int[nGramLen];
		n0=new double[nGramLen];
		for(int i=0;i<nGramLen;i++){
			frequencyMatrix.add(new Hashtable<String,Integer>());
			histogram.add(new Hashtable<Integer,Integer>());
		}

		//Building the LM
		BufferedReader readbuffer = new BufferedReader(new InputStreamReader(new FileInputStream(trainFile)));
		String line;
		StringBuilder tmpStr;
		ArrayList<String> words;
		int totalWords=0;
		while ((line=readbuffer.readLine())!=null){
			line=preprocessingStep(line);			
			if(!line.isEmpty()){
				words = new ArrayList<String>(Arrays.asList(line.split("\\s+")));
				words.removeAll(Arrays.asList("", null));
				totalWords+=words.size();

				for(int i=0;i<=words.size()-nGramLen;i++){
					int startIndx=i;
					int endIndx=startIndx+nGramLen-1;
					tmpStr=new StringBuilder();
					for(int j=startIndx;j<=endIndx;j++){
						tmpStr.append(words.get(j)+" ");
						int ngSize=tmpStr.toString().trim().split(" ").length-1;

						Hashtable<String,Integer> tmpTable=frequencyMatrix.get(ngSize); 
						int count;
						if(tmpTable.containsKey(tmpStr.toString().trim())){
							count=tmpTable.get(tmpStr.toString().trim()).intValue()+1;
						}else{
							count=1;
						}
						tmpTable.put(tmpStr.toString().trim(),new Integer(count));
						frequencyMatrix.set(ngSize,tmpTable);				
					}				
				}		
			}
		}		
		readbuffer.close();	

		totalTokens[0]=totalWords;
		for(int i=1;i<totalTokens.length;i++){
			totalTokens[i]=totalTokens[i-1]-1;
		}

		for(int i=0;i<nGramLen;i++){
			ArrayList<Integer> nGramsFreq=new ArrayList<Integer>(frequencyMatrix.get(i).values());
			bldHistogram(nGramsFreq,histogram.get(i));
		}

		double unseen;
		int vocab= frequencyMatrix.get(0).size();
		for(int i=0;i<nGramLen;i++){
			unseen = Math.pow(vocab, (i+1))- frequencyMatrix.get(i).size();
			if(unseen==0){
				unseen=1000000;
			}
			n0[i]=unseen;
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


	private void bldHistogram(ArrayList<Integer> freqs,Hashtable<Integer,Integer>hisogram){
		for(int i=0;i<freqs.size();i++){
			int counts =1;
			if(hisogram.containsKey(freqs.get(i))){
				counts+=hisogram.get(freqs.get(i));
			}
			hisogram.put(freqs.get(i), counts);
		}

	}


	private double gtSmoothing(String nGramStr){
		int nGramSze=nGramStr.split("\\s+").length;
		int wrkCount =0;
		if(frequencyMatrix.get(nGramSze-1).containsKey(nGramStr)){
			wrkCount=frequencyMatrix.get(nGramSze-1).get(nGramStr);
		}
		double nc=1;
		double nc1=0;
		nc = (wrkCount == 0 ) ? n0[nGramSze-1]:(double)histogram.get(nGramSze-1).get(new Integer(wrkCount)).intValue();

		if(histogram.get(nGramSze-1).containsKey(new Integer(wrkCount+1))){
			nc1=(double)histogram.get(nGramSze-1).get(new Integer(wrkCount+1)).intValue();
		}
		if(nc1==0 || wrkCount>maxSmoohing){
			double numeratorCounts=wrkCount;
			double denominatorCounts=0;
			if(nGramSze==1){
				denominatorCounts=totalTokens[nGramSze-1];
			}else{
				String[]grams=nGramStr.split("\\s+");
				StringBuilder denominatorStr= new StringBuilder();
				for(int i=0; i<nGramSze-1;i++){
					denominatorStr.append(grams[i]+" ");
				}
				denominatorCounts=frequencyMatrix.get(nGramSze-2).get(denominatorStr.toString().trim()).intValue();
			}
			return(numeratorCounts/denominatorCounts);			
		}else{
			int tokenCount=totalTokens[nGramSze-1];
			if(wrkCount==0){
				return (double)((wrkCount+1)*nc1)/(double)(tokenCount*nc);
			}else{
				double nk=1;
				if(histogram.get(nGramSze-1).containsKey(new Integer(maxSmoohing+1))){
					nk=(double)histogram.get(nGramSze-1).get(new Integer(maxSmoohing+1)).intValue();
				}
				double ng1=1;
				ng1=(double)histogram.get(nGramSze-1).get(new Integer(1)).intValue();
				double cstr=((wrkCount+1)*nc1)/nc;
				cstr=cstr-((wrkCount*(maxSmoohing+1)*nk)/ng1);
				cstr=cstr/(1-(((maxSmoohing+1)*nk)/ng1));
				return (cstr/tokenCount);				
			}

		}		
	}

	public double getSentenceProbability(String str){
		str=preprocessingStep(str);
		if(str.isEmpty()){
			return Double.NEGATIVE_INFINITY;
		}
		ArrayList<String> words = new ArrayList<String>(Arrays.asList(str.split("\\s+")));
		words.removeAll(Arrays.asList("", null));
		double probabiliy=Math.log10(gtSmoothing(words.get(0)));
		for(int i=0;i<=words.size()-nGramLen;i++){
			int startIndx=i;
			int endIndx=startIndx+nGramLen-1;
			StringBuilder currStr=new StringBuilder();
			for(int j=startIndx;j<=endIndx;j++){
				currStr.append(words.get(j)+" ");
			}
			double currProbability=gtSmoothing(currStr.toString().trim());
			if(currProbability==0||currProbability==Double.POSITIVE_INFINITY){
				probabiliy+=Double.NEGATIVE_INFINITY;
			}else{
				probabiliy+=Math.log10(currProbability);
			}						
		}		
		return probabiliy;		
	}

	private void finalOutputLM(String outfile) throws IOException{
		ObjectOutput output = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(outfile)));
		output.writeObject(lm);
		output.writeObject(nGramLen);
		output.writeObject(frequencyMatrix);
		output.writeObject(totalTokens);
		output.writeObject(histogram);
		output.writeObject(n0);
		output.writeObject(maxSmoohing);
		output.close();		
	}


}
