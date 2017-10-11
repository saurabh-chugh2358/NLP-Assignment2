package TriGramWordLangId;
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
import java.util.Set;




public class TrigramWordLangIdKBO {
	private ArrayList<Hashtable<String,Integer>>frequencyMatrix=new ArrayList<Hashtable<String,Integer>>();
	private ArrayList<Hashtable<Integer,Integer>>histogram=new ArrayList<Hashtable<Integer,Integer>>();
	private int[] totalNoTokens;
	private double[] n0;
	private int k=5;
	private Hashtable<String,Double>backOffParameters=new Hashtable<String,Double>();	

	private int nGramLen;
	private String unKnwn="@@UNKNOWN@@@";
	private String LmType="KBO";


	public static void main(String[] args) throws IOException {
		String[] trainingFiles = new String[]{"HW2-english.txt", "HW2-french.txt", "HW2-german.txt"};
		String[] langArray = new String[]{"EN", "FR", "GR"};	
		Hashtable<String,Integer>lang2Idx=new Hashtable<String,Integer>();
		lang2Idx.put("EN", 0);
		lang2Idx.put("FR", 1);
		lang2Idx.put("GR", 2);
		String[] outLMs = new String[]{"EN_Tri_Words_KBO.bin", "FR_Tri_Words_KBO.bin", "GR_Tri_Words_KBO.bin"};

		int maxNgramSize=3;

		//Building the LMs
		TrigramWordLangIdKBO[] lms=new TrigramWordLangIdKBO[trainingFiles.length];
		for(int i=0;i<lms.length;i++){
			lms[i]=new TrigramWordLangIdKBO(trainingFiles[i],outLMs[i],maxNgramSize);
		}

		//Load the gold-Id file
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

		//Load the test file
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

		//Find the Best tag for each sentence
		OutputStream outputFile=new FileOutputStream("TrigramWordLangId-KBO.out");
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(outputFile, "UTF-8"));
		out.write("ID LANG\n");//write the header
		double finalScore=0;
		String calcId=langArray[0];
		int correct=0;
		int[][]confusionMatrix = new int[langArray.length][langArray.length];
		for(int i=0;i<langArray.length;i++){
			for(int j=0;j<langArray.length;j++){
				confusionMatrix[i][j]=0;
			}
		}

		for(int i=0;i<tstString.size();i++){
			finalScore=lms[0].getSentenceProbability(tstString.get(i));
			calcId=langArray[0];
			for(int j=1;j<lms.length;j++){
				double tmpScore=lms[j].getSentenceProbability(tstString.get(i));
				if(tmpScore>finalScore){
					finalScore=tmpScore;
					calcId=langArray[j];
				}
			}
			out.write(i+". "+calcId+"\n");
			if(calcId.equals(goldLangIDs.get(i))){
				correct++;
			}else{
				confusionMatrix[lang2Idx.get(goldLangIDs.get(i))][lang2Idx.get(calcId)]++;
			}
		}

		double accuracy=(double)correct/(double)tstString.size();
		out.write("\nAccuracy= "+Double.toString(accuracy));
		out.close();

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

	public TrigramWordLangIdKBO(String trainFile, String outputBinaryLmFile, int nGramLen) throws IOException{
		this.nGramLen=nGramLen;
		totalNoTokens=new int[nGramLen];
		n0=new double[nGramLen];
		for(int i=0;i<nGramLen;i++){
			Hashtable<String,Integer> tmpFreq=new Hashtable<String,Integer>();
			frequencyMatrix.add(tmpFreq);
			Hashtable<Integer,Integer> tmpHist=new Hashtable<Integer,Integer>();
			histogram.add(tmpHist);
		}

		//Building the LM
		Hashtable<String,Hashtable<String,Integer>>highGram=new Hashtable<String,Hashtable<String,Integer>>();
		BufferedReader readbuffer = new BufferedReader(new InputStreamReader(new FileInputStream(trainFile)));
		String line;
		int totalWords=0;
		while ((line=readbuffer.readLine())!=null){
			line=preprocessingStep(line);			
			if(!line.isEmpty()){
				ArrayList<String> words = new ArrayList<String>(Arrays.asList(line.split("\\s+")));
				words.removeAll(Arrays.asList("", null));
				totalWords+=words.size();

				for(int i=0;i<=words.size()-nGramLen;i++){
					int startIndx=i;
					int endIndx=startIndx+nGramLen-1;
					StringBuilder tmpStr=new StringBuilder();

					String lGram = null , hghGram=null;
					for(int j=startIndx;j<=endIndx;j++){
						tmpStr.append(words.get(j)+" ");
						int ngSize=tmpStr.toString().trim().split(" ").length-1;

						Hashtable<String,Integer> tmpTable=frequencyMatrix.get(ngSize); 
						int count = tmpTable.containsKey(tmpStr.toString().trim()) ? tmpTable.get(tmpStr.toString().trim()).intValue()+1 : 1;

						tmpTable.put(tmpStr.toString().trim(),new Integer(count));
						frequencyMatrix.set(ngSize,tmpTable);
						if(lGram==null){
							if(tmpStr.toString().trim().split(" ").length<this.nGramLen && !highGram.containsKey(tmpStr.toString().trim())){
								lGram=tmpStr.toString().trim();
								highGram.put(tmpStr.toString().trim(), new Hashtable<String,Integer>());								
							}						
						}else{
							hghGram=tmpStr.toString().trim();
							Hashtable<String,Integer>hGram;
							if(!highGram.containsKey(lGram)){
								hGram=new Hashtable<String,Integer>();
							}else{
								hGram=highGram.get(lGram);
							}						
							hGram.put(hghGram,1);
							highGram.put(lGram, hGram);
							if(tmpStr.toString().trim().split(" ").length>=this.nGramLen){
								continue;
							}else{
								lGram=hghGram;	
								if(!highGram.containsKey(lGram)){
									Hashtable<String,Integer>tmpHGram=new Hashtable<String,Integer>();
									highGram.put(lGram, tmpHGram);							
								}
							}						
						}
					}				
				}		
			}
		}		
		readbuffer.close();	

		totalNoTokens[0]=totalWords;
		for(int i=1;i<totalNoTokens.length;i++){
			totalNoTokens[i]=totalNoTokens[i-1]-1;
		}

		//Calculating good turing parameters
		for(int i=0;i<nGramLen;i++){
			ArrayList<Integer> freqList=new ArrayList<Integer>(frequencyMatrix.get(i).values());
			bldHistogram(freqList,histogram.get(i));
		}
		int vocab= frequencyMatrix.get(0).size();
		for(int i=0;i<nGramLen;i++){
			double unseen=Math.pow((double)vocab, (double)(i+1))-(double)frequencyMatrix.get(i).size();
			if(unseen==0){
				unseen=1000000;
			}
			n0[i]=unseen;
		}

		backOffParameters.put(unKnwn, 1.0);
		Set<String>lGrams= highGram.keySet();
		for(String lGram: lGrams){
			Set<String>hGramSet=highGram.get(lGram).keySet();
			double beta1=1;
			double beta2=1;
			for(String higherGram: hGramSet){
				beta1=beta1-calcKatzProb(higherGram);//beta1-p(Wn|W1..n-1)
				beta2=beta2-calcKatzProb(higherGram.split(" ",2)[1]);//beta2-p(Wn|W2..n-1)        		 
			}
			if(beta1==0){
				beta1=0.00000001;//to avoid 0 back-off score
			}
			if(beta2==0){
				beta2=0.00000001;//to avoid infinity back-off score
			}
			double alpha=beta1/beta2;
			backOffParameters.put(lGram, new Double(alpha));	           
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

	public double getSentenceProbability(String str){
		str=preprocessingStep(str);
		if(str.isEmpty()){
			return Double.NEGATIVE_INFINITY;
		}
		ArrayList<String> words = new ArrayList<String>(Arrays.asList(str.split("\\s+")));
		words.removeAll(Arrays.asList("", null));
		double probabiliy=Math.log10(nGramProbability(words.get(0)));
		for(int i=0;i<=words.size()-nGramLen;i++){
			int startIndx=i;
			int endIndx=startIndx+nGramLen-1;
			StringBuilder runningStr=new StringBuilder();
			for(int j=startIndx;j<=endIndx;j++){
				runningStr.append(words.get(j)+" ");
			}
			double currProbability=nGramProbability(runningStr.toString().trim());
			if(currProbability==0||currProbability==Double.POSITIVE_INFINITY){
				probabiliy+=Double.NEGATIVE_INFINITY;
			}else{
				probabiliy+=Math.log10(currProbability);
			}						
		}		
		return probabiliy;		
	}

	private double nGramProbability(String nGramStr){
		int nGramLength=nGramStr.split("\\s+").length;		
		if(frequencyMatrix.get(nGramLength-1).containsKey(nGramStr)){
			return calcKatzProb(nGramStr);
		}else{
			if(nGramLength==1){
				int tokens=totalNoTokens[nGramLength-1];				
				return (getGTDiscount(nGramStr)/(double)tokens);
			}
			double theta=calsTheta(nGramStr);
			return (theta*nGramProbability(nGramStr.split(" ",2)[1]));		
		}	
	}

	private double calcKatzProb(String nGramStr){
		int nGramSze=nGramStr.split("\\s+").length;		
		if(frequencyMatrix.get(nGramSze-1).containsKey(nGramStr)){
			double c_star=getGTDiscount(nGramStr);
			double denominatorCounts=0;
			if(nGramSze==1){
				denominatorCounts=totalNoTokens[nGramSze-1];
			}else{
				String[]grams=nGramStr.split("\\s+");
				StringBuilder denominatorStr= new StringBuilder();
				for(int i=0; i<nGramSze-1;i++){
					denominatorStr.append(grams[i]+" ");
				}
				denominatorCounts=frequencyMatrix.get(nGramSze-2).get(denominatorStr.toString().trim()).intValue();
			}
			return (c_star/denominatorCounts);
		}else{
			return 0;
		}	
	}

	private double getGTDiscount(String nGramStr){
		int nGramSze=nGramStr.split("\\s+").length;
		int c =0;
		if(frequencyMatrix.get(nGramSze-1).containsKey(nGramStr)){
			c=frequencyMatrix.get(nGramSze-1).get(nGramStr);
		}
		double N_C=1;
		if(c!=0){
			N_C=(double)histogram.get(nGramSze-1).get(new Integer(c)).intValue();
		}else{
			N_C=n0[nGramSze-1];
		}
		double N_C_plus_1=0;
		if(histogram.get(nGramSze-1).containsKey(new Integer(c+1))){
			N_C_plus_1=(double)histogram.get(nGramSze-1).get(new Integer(c+1)).intValue();
		}
		if(N_C_plus_1==0 || c>k){
			return(c);			
		}else{
			if(c==0){
				return (double)((c+1)*N_C_plus_1)/(double)(N_C);
			}else{
				double N_K_plus_1=1;
				if(histogram.get(nGramSze-1).containsKey(new Integer(k+1))){
					N_K_plus_1=(double)histogram.get(nGramSze-1).get(new Integer(k+1)).intValue();
				}
				double N_1=1;
				N_1=(double)histogram.get(nGramSze-1).get(new Integer(1)).intValue();
				double c_star=((c+1)*N_C_plus_1)/N_C;
				c_star=c_star-((c*(k+1)*N_K_plus_1)/N_1);
				c_star=c_star/(1-(((k+1)*N_K_plus_1)/N_1));
				return (c_star);				
			}

		}		
	}

	private double calsTheta(String nGramStr) {
		int nGramSze=nGramStr.split("\\s+").length;
		if(frequencyMatrix.get(nGramSze-1).containsKey(nGramStr)){
			return 0.0;
		}else{
			String backOffStr=nGramStr.replaceAll(" [^ ]+$", "").trim();
			return getBackOff(backOffStr);
		}
	}

	private double getBackOff(String nGramStr) {
		if(backOffParameters.containsKey(nGramStr)){
			return backOffParameters.get(nGramStr);
		}else{
			return backOffParameters.get(unKnwn);
		}
	}

	private void finalOutputLM(String outfile) throws IOException{
		OutputStream file = new FileOutputStream(outfile);
		OutputStream buffer = new BufferedOutputStream(file);
		ObjectOutput output = new ObjectOutputStream(buffer);
		output.writeObject(LmType);
		output.writeObject(nGramLen);
		output.writeObject(frequencyMatrix);
		output.writeObject(totalNoTokens);
		output.writeObject(histogram);
		output.writeObject(n0);
		output.writeObject(k);
		output.writeObject(backOffParameters);
		output.writeObject(unKnwn);
		output.close();		
	}

}
