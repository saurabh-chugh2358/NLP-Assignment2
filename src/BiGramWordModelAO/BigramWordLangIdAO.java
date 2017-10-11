package BiGramWordModelAO;
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

public class BigramWordLangIdAO {
	private int[] totalTokens;
	private String lm="AddOneSmoothing";
	private int nGramLen;
	private ArrayList<Hashtable<String,Integer>>frequencyMatrix=new ArrayList<Hashtable<String,Integer>>();


	public static void main(String[] args) throws IOException {
		int nGramLength=2;
		String[] trainingFiles = new String[]{"HW2-english.txt", "HW2-french.txt", "HW2-german.txt"};
		String[] langArray = new String[]{"EN", "FR", "GR"};
		Hashtable<String,Integer>languageIndx=new Hashtable<String,Integer>();
		languageIndx.put("EN", 0);
		languageIndx.put("FR", 1);
		languageIndx.put("GR", 2);
		String[] outputLMs = new String[]{"EN_Bi_Words_AO.bin", "FR_Bi_Words_AO.bin", "GR_Bi_Words_AO.bin"};

		BigramWordLangIdAO[] langModels=new BigramWordLangIdAO[trainingFiles.length];
		for(int i=0;i<langModels.length;i++){
			langModels[i]=new BigramWordLangIdAO(trainingFiles[i], outputLMs[i], nGramLength);
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

		BufferedWriter finalOuput = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("BigramLetterLangId-AO.out"), "UTF-8"));
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
				confusionMatrix[languageIndx.get(goldLangIDs.get(i))][languageIndx.get(calcId)]++;
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


	//n-grams with any size using add one smoothing
	public BigramWordLangIdAO(String trainFile, String outputBinaryLmFile, int nGramLen) throws IOException{
		this.nGramLen=nGramLen;
		totalTokens=new int[nGramLen];
		for(int i=0;i<nGramLen;i++){
			frequencyMatrix.add(new Hashtable<String,Integer>());
		}

		//Building the LM
		BufferedReader readbuffer = new BufferedReader(new InputStreamReader(new FileInputStream(trainFile)));
		String line;
		int totalLeters=0;
		ArrayList<String> letters;
		while ((line=readbuffer.readLine())!=null){
			line=preprocessingStep(line);			
			if(!line.isEmpty()){
				letters = new ArrayList<String>(Arrays.asList(line.split("\\s+")));
				letters.removeAll(Arrays.asList("", null));
				totalLeters+=letters.size();


				for(int i=0;i<=letters.size()-nGramLen;i++){
					int starIndx=i;
					int endIndx=starIndx+nGramLen-1;
					StringBuilder tmpStr=new StringBuilder();
					for(int j=starIndx;j<=endIndx;j++){
						tmpStr.append(letters.get(j)+" ");
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

		totalTokens[0]=totalLeters;
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

	private double aoSmoothing(String numeratoStr, String denominatorStr){
		int nGramSze=numeratoStr.split("\\s+").length;
		double numeratorCounts=1;
		double denominatorCounts=0;
		if(frequencyMatrix.get(nGramSze-1).containsKey(numeratoStr)){
			numeratorCounts+=frequencyMatrix.get(nGramSze-1).get(numeratoStr);
		}

		if(frequencyMatrix.get(nGramSze-2).containsKey(denominatorStr)){
			denominatorCounts=frequencyMatrix.get(nGramSze-2).get(denominatorStr);
		}
		denominatorCounts+=frequencyMatrix.get(nGramSze-2).size();
		return (numeratorCounts/denominatorCounts);


	}


	public double getSentenceProbability(String str){
		str=preprocessingStep(str);
		if(str.isEmpty()){
			return Double.NEGATIVE_INFINITY;
		}
		ArrayList<String> words = new ArrayList<String>(Arrays.asList(str.split("\\s+")));
		words.removeAll(Arrays.asList("", null));
		double probabiliy=Math.log10(nGramProbability(words.get(0)));
		for(int i=0;i<=words.size()-nGramLen;i++){
			int startIdx=i;
			int endIdx=startIdx+nGramLen-1;
			StringBuilder currStr=new StringBuilder();
			for(int j=startIdx;j<=endIdx;j++){
				currStr.append(words.get(j)+" ");
			}
			double currProbability=nGramProbability(currStr.toString().trim());
			if(currProbability==0||currProbability==Double.POSITIVE_INFINITY){
				probabiliy+=Double.NEGATIVE_INFINITY;
			}else{
				probabiliy+=Math.log10(currProbability);
			}						
		}		
		return probabiliy;		
	}

	private double nGramProbability(String ngramStr){
		String[]grams=ngramStr.split("\\s+");
		int nGramSze=grams.length;
		if(nGramSze>1){
			StringBuilder denominatorStr= new StringBuilder();
			for(int i=0; i<nGramSze-1;i++){
				denominatorStr.append(grams[i]+" ");
			}			
			return aoSmoothing(ngramStr,denominatorStr.toString().trim());
		}else{
			double numeratorCounts=1;
			double denominatorCounts=totalTokens[nGramSze-1]+frequencyMatrix.get(nGramSze-1).size();//N+|v|
			if(frequencyMatrix.get(nGramSze-1).containsKey(ngramStr)){
				numeratorCounts+=frequencyMatrix.get(nGramSze-1).get(ngramStr);
			}
			return (numeratorCounts/denominatorCounts);

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
