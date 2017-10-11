import BiGramLetterModel.*;
import BiGramWordModelAO.*;
import BiGramWordModelGT.*;
import TriGramWordLangId.*;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;

/**
 * Assignment-2 Q6
 *
 */

public class Perplexity {
    //Generic parameters
    private ArrayList<Hashtable<String,Integer>>freq=new ArrayList<Hashtable<String,Integer>>();
    private int[] totalNoTokens;
    private int maxNgramSize;
    
    //Good-Turing specific parameters
    private ArrayList<Hashtable<Integer,Integer>>histogram=new ArrayList<Hashtable<Integer,Integer>>();//used for Good-Turing smoothing
    private double[] N_0;//the number of unseen grams for each n-gram size
    private int k;//this is the max counts that can be smoothed using Good-Turing as mentioned in Jurafsky and Martin Book.
    //Above that number, use MLE
    
    //Katz Back-Off specific parameters
    private Hashtable<String,Double>alphas=new Hashtable<String,Double>();
    private String unk;//unknow
    
    private enum SmoothingType{noSmoothing, AddOne, GoodTuring, KBO}
    private SmoothingType smT;
    
    /**
     * @throws IOException
     * @throws ClassNotFoundException
     *
     */
    @SuppressWarnings("unchecked")
    public Perplexity(String LM) throws IOException, ClassNotFoundException {
        InputStream is=new FileInputStream(LM);
        InputStream buffer = new BufferedInputStream(is);
        ObjectInput input = new ObjectInputStream (buffer);
        smT=SmoothingType.valueOf((String)input.readObject());
        maxNgramSize=(int)input.readObject();
        freq=(ArrayList<Hashtable<String,Integer>>)input.readObject();
        totalNoTokens=(int[])input.readObject();
        
        if(smT==SmoothingType.GoodTuring||smT==SmoothingType.KBO){//both methods use these parameters
            histogram=(ArrayList<Hashtable<Integer,Integer>>)input.readObject();
            N_0=(double[])input.readObject();
            k=(int)input.readObject();
        }
        
        if(smT==SmoothingType.KBO){
            alphas=(Hashtable<String,Double>)input.readObject();
            unk=(String)input.readObject();
        }
        
        input.close();
    }
    /**
     * Calculate the LM log perplexity using the input test file.
     * The function uses the perplexity formula in JM3(4)
     * @param TestFile
     * @return
     * @throws IOException
     */
    public double calculateLogPerplexity(String TestFile) throws IOException{
        //Load the test file
        InputStream testFileIS=new FileInputStream(TestFile);
        BufferedReader readbuffer = new BufferedReader(new InputStreamReader(testFileIS));
        //ArrayList<String>testSentences=new ArrayList<String>();
        String strRead=readbuffer.readLine();
        int totalTokens=0;
        double logPerplexity=0;
        while (strRead!=null){
            strRead=strRead.trim();
            if(strRead.isEmpty()){//to skip empty lines
                strRead=readbuffer.readLine();
                continue;
            }
            totalTokens+=getTokensNo(strRead.replaceAll("^\\d+.\\s", "")); //remove the line index before calculating the number of tokens
            logPerplexity+=getSentProb(strRead.replaceAll("^\\d+.\\s", ""));//remove the line index before calculating the sentence probability
            strRead=readbuffer.readLine();
        }
        readbuffer.close();
        logPerplexity=logPerplexity*(-1/(double)totalTokens);
        return logPerplexity;
    }
    /**
     * This function returns the probability of the input sentence in log space
     */
    public double getSentProb(String sentence){
        sentence=preprocess(sentence);
        if(sentence.isEmpty()){
            return Double.NEGATIVE_INFINITY;
        }
        ArrayList<String> letters=null;
        if(smT==SmoothingType.noSmoothing){//i.e. letter-grams
            letters= new ArrayList<String>(Arrays.asList(sentence.replaceAll("", "\t").split("\t")));
        }else{//i.e. word-grams
            letters= new ArrayList<String>(Arrays.asList(sentence.replaceAll("\\s+", "\t").split("\t")));
        }
        letters.removeAll(Arrays.asList("", null));// cleaning the array by removing any empty or null elements
        
        double logProb=Math.log10(getNgramProb(letters.get(0)));
        
        for(int i=0;i<=letters.size()-maxNgramSize;i++){
            int startIdx=i;
            int endIdx=startIdx+maxNgramSize-1;
            StringBuilder runningStr=new StringBuilder();
            for(int j=startIdx;j<=endIdx;j++){
                runningStr.append(letters.get(j)+" ");
            }
            double tmpProb=getNgramProb(runningStr.toString().trim());
            if(tmpProb==0||tmpProb==Double.POSITIVE_INFINITY){
                logProb+=Double.NEGATIVE_INFINITY;//to decrease the prob of this sentence
            }else{
                logProb+=Math.log10(tmpProb);
            }
        }
        return logProb;
    }
    /**
     * This function removes the numbers and the punctuation from the input sentence because they are not an
     * indicator of any language. I only keep the single quote because it can be a signal for English and French
     * The function also convert all letters to lower case
     * @param str
     * @return
     */
    private String preprocess(String str){
        String punc="([\"\'\\}\\{\\«\\»\\-\\=\\\"\\_\\:\\#\\@\\!\\?\\^\\/\\(\\)\\[\\]\\%\\;\\\\\\+\\.\\,]+)";
        String nums="(\\d+)";
        return str.replaceAll(punc, " $1 ").replaceAll(nums, " $1 ").replaceAll("\\s+", " ").trim().toLowerCase();
    }
    /**
     * returns the number of tokens in the input sentence based on the KM type.
     * If letter-grams, it returns the number of characters.
     * If word-grams, it returns the number of words.
     * @param sentence
     * @return
     */
    private int getTokensNo(String sentence){
        if(smT==SmoothingType.noSmoothing){//i.e. letter-grams
            return preprocess(sentence).split("").length;
        }else{//i.e. word-grams
            return preprocess(sentence).split("\\s+").length;
        }
    }
    /**
     * It returns the probability of the input nGram based on the type of the input LM
     * @param ngramStr
     * @return
     */
    private double getNgramProb(String ngramStr){
        if(smT==SmoothingType.noSmoothing){
            return getNoSmoothingProb(ngramStr);
        }else if(smT==SmoothingType.AddOne){
            return getAddOneSmoothing(ngramStr);
        }else if(smT==SmoothingType.GoodTuring){
            return getGTSmoothing(ngramStr);
        }else{//i.e. Katz Back-Off
            return getKBOProb(ngramStr);
        }
        
    }
    
    /**
     * Calculate the probability of the input nGram without smoothing
     * It uses this equation if the input string is >= 2 grams:
     * P(w_n|w_n-1) = C(w_n-1w_n)/C(w_n-1)
     * But if the input is only uni-gram:
     * P(w_n) = C(w_n)/|V|; where |V| is the number of uni-grams in the LM
     * @param ngramStr
     * @return
     */
    private double getNoSmoothingProb(String ngramStr){
        String[]grams=ngramStr.split("\\s+");
        int nGramSze=grams.length;
        double numeratorCounts=0;
        double denominatorCounts=0;
        
        if(freq.get(nGramSze-1).containsKey(ngramStr)){// I used -1 because the unigram is stored on freq[0], bi-gram is stored in freq[1], ...etc
            numeratorCounts=freq.get(nGramSze-1).get(ngramStr);
        }
        if(nGramSze>1){//i.e. > uni-gram
            StringBuilder denominatorStr= new StringBuilder();
            for(int i=0; i<nGramSze-1;i++){
                denominatorStr.append(grams[i]+" ");
            }
            if(freq.get(nGramSze-2).containsKey(denominatorStr.toString().trim())){
                denominatorCounts=freq.get(nGramSze-2).get(denominatorStr.toString().trim());
            }
        }else{//i.e. the input is uni-gram
            denominatorCounts=totalNoTokens[0];
        }
        if(denominatorCounts==0){
            return Double.POSITIVE_INFINITY;
        }else{
            return (numeratorCounts/denominatorCounts);
        }
    }
    /**
     * This function calculate the probability of the input n-gram string with add one smoothing using this equation if the input string is >= 2 grams:
     * P(w_n|w_n-1) = (C(w_n-1w_n)+1)/(C(w_n-1)+|v|)
     * But if the input is only uni-gram:
     * P(w_n) = (C(w_n)+)/|V|; where |V| is the number of uni-grams in the LM
     * @param ngramStr
     * @return
     */
    private double getAddOneSmoothing(String ngramStr){
        String[]grams=ngramStr.split("\\s+");
        int nGramSze=grams.length;
        if(nGramSze>1){//i.e. > uni-gram
            StringBuilder denominatorStr= new StringBuilder();
            for(int i=0; i<nGramSze-1;i++){
                denominatorStr.append(grams[i]+" ");
            }
            double numeratorCounts=1;
            double denominatorCounts=0;
            if(freq.get(nGramSze-1).containsKey(ngramStr)){// I used -1 because the unigram is stored on freq[0], bi-gram is stored in freq[1], ...etc
                numeratorCounts+=freq.get(nGramSze-1).get(ngramStr);
            }
            if(freq.get(nGramSze-2).containsKey(denominatorStr)){// I used -1 because the unigram is stored on freq[0], bi-gram is stored in freq[1], ...etc
                denominatorCounts=freq.get(nGramSze-2).get(denominatorStr);
            }
            denominatorCounts+=freq.get(nGramSze-2).size();
            return (numeratorCounts/denominatorCounts);
        }else{//i.e. the input is uni-gram
            double numeratorCounts=1;
            double denominatorCounts=totalNoTokens[nGramSze-1]+freq.get(nGramSze-1).size();//N+|v|
            if(freq.get(nGramSze-1).containsKey(ngramStr)){// I used -1 because the unigram is stored on freq[0], bi-gram is stored in freq[1], ...etc
                numeratorCounts+=freq.get(nGramSze-1).get(ngramStr);
            }
            return (numeratorCounts/denominatorCounts);
            
        }
    }
    /**
     * this function uses Good-Turing smoothing technique. It takes an n-gram strings and returns its smoothed frequency using this formula:
     * P(w_n,w_n-1) = (c+1)(N_c+1)/(N*N_c)
     * If N_c+1=0 --> use the MLE formula: P(w_n,w_n-1) = C(w_n-1w_n)/C(w_n-1)
     * If N_c=0 --> P(w_n,w_n-1) = (c+1)(N_c+1)/(N)
     */
    private double getGTSmoothing(String ngramStr){
        int nGramSze=ngramStr.split("\\s+").length;
        int c =0;
        if(freq.get(nGramSze-1).containsKey(ngramStr)){// I used -1 because the unigram is stored on freq[0], bi-gram is stored in freq[1], ...etc
            c=freq.get(nGramSze-1).get(ngramStr);
        }
        double N_C=1;
        if(c!=0){
            N_C=(double)histogram.get(nGramSze-1).get(new Integer(c)).intValue();
        }else{//get the N_0
            N_C=N_0[nGramSze-1];
        }
        double N_C_plus_1=0;
        if(histogram.get(nGramSze-1).containsKey(new Integer(c+1))){// I used -1 because the unigram is stored on freq[0], bi-gram is stored in freq[1], ...etc
            N_C_plus_1=(double)histogram.get(nGramSze-1).get(new Integer(c+1)).intValue();
        }
        if(N_C_plus_1==0 || c>k){//i.e. this is the most frequent n-gram or the n-gram counts is higher than
            // the maximum allowed smoothing frequency as stated in Jurafsky and Martin Book.
            // then use MLE
            double numeratorCounts=c;
            double denominatorCounts=0;
            if(nGramSze==1){//i.e. uni-gram
                denominatorCounts=totalNoTokens[nGramSze-1];
            }else{//i.e. higher grams
                String[]grams=ngramStr.split("\\s+");
                StringBuilder denominatorStr= new StringBuilder();
                for(int i=0; i<nGramSze-1;i++){
                    denominatorStr.append(grams[i]+" ");
                }
                denominatorCounts=freq.get(nGramSze-2).get(denominatorStr.toString().trim()).intValue();
            }
            return(numeratorCounts/denominatorCounts);
        }else{//i.e. use GT- smoothing formula
            int N=totalNoTokens[nGramSze-1];
            if(c==0){
                return (double)((c+1)*N_C_plus_1)/(double)(N*N_C);
            }else{
                double N_K_plus_1=1;
                if(histogram.get(nGramSze-1).containsKey(new Integer(k+1))){// I used -1 because the unigram is stored on freq[0], bi-gram is stored in freq[1], ...etc
                    N_K_plus_1=(double)histogram.get(nGramSze-1).get(new Integer(k+1)).intValue();
                }
                double N_1=1;
                N_1=(double)histogram.get(nGramSze-1).get(new Integer(1)).intValue();
                double c_star=((c+1)*N_C_plus_1)/N_C;
                c_star=c_star-((c*(k+1)*N_K_plus_1)/N_1);
                c_star=c_star/(1-(((k+1)*N_K_plus_1)/N_1));
                return (c_star/N);
            }
        }
        
    }
    
    
    /**
     * This function implements the Katz Back-Off as described in Jurafsky book
     * @param ngramStr
     * @return
     */
    private double getKBOProb(String ngramStr){
        int nGramSze=ngramStr.split("\\s+").length;
        if(freq.get(nGramSze-1).containsKey(ngramStr)){// I used -1 because the unigram is stored on freq[0], bi-gram is stored in freq[1], ...etc
            return getKatzProb(ngramStr);//p(Wn|W1...n-1)
        }else{//i.e. do back off
            if(nGramSze==1){//i.e. unigram
                int N=totalNoTokens[nGramSze-1];
                return (getGTDiscount(ngramStr)/(double)N);
            }
            double theta=getTheta(ngramStr);
            return (theta*getNgramProb(ngramStr.split(" ",2)[1]));//p(Wn|W2..n-1)
        }
    }
    /**
     * This function calculate the probability of the input n-gram string with Katz discounted probability:
     * P(w_n,w_n-1) = C_star(w_n-1w_n)/C(w_n-1)
     * where C_star is the discounted frequency using Good-Turing smoothing
     *
     * @param ngramStr
     * @return
     */
    private double getKatzProb(String nGramStr){
        int nGramSze=nGramStr.split("\\s+").length;
        if(freq.get(nGramSze-1).containsKey(nGramStr)){// I used -1 because the unigram is stored on freq[0], bi-gram is stored in freq[1], ...etc
            double c_star=getGTDiscount(nGramStr);
            double denominatorCounts=0;
            if(nGramSze==1){//i.e. uni-gram
                denominatorCounts=totalNoTokens[nGramSze-1];
            }else{//i.e. higher grams
                String[]grams=nGramStr.split("\\s+");
                StringBuilder denominatorStr= new StringBuilder();
                for(int i=0; i<nGramSze-1;i++){
                    denominatorStr.append(grams[i]+" ");
                }
                denominatorCounts=freq.get(nGramSze-2).get(denominatorStr.toString().trim()).intValue();
            }
            return (c_star/denominatorCounts);//discounted counts/lower model counts
        }else{
            return 0;
        }
    }
    
    /**
     * this function applies the Good-Turing smoothing technique. It takes an n-gram strings and returns its discounted counts using this formula:
     * C(w_n,w_n-1) = (c+1)(N_c+1)/(N_c)
     * If N_c+1=0 --> use the MLE formula: c(w_n,w_n-1) = C(w_n-1w_n)
     */
    private double getGTDiscount(String nGramStr){
        int nGramSze=nGramStr.split("\\s+").length;
        int c =0;
        if(freq.get(nGramSze-1).containsKey(nGramStr)){// I used -1 because the unigram is stored on freq[0], bi-gram is stored in freq[1], ...etc
            c=freq.get(nGramSze-1).get(nGramStr);
        }
        double N_C=1;
        if(c!=0){
            N_C=(double)histogram.get(nGramSze-1).get(new Integer(c)).intValue();
        }else{//get the N_0
            N_C=N_0[nGramSze-1];
        }
        double N_C_plus_1=0;
        if(histogram.get(nGramSze-1).containsKey(new Integer(c+1))){// I used -1 because the unigram is stored on freq[0], bi-gram is stored in freq[1], ...etc
            N_C_plus_1=(double)histogram.get(nGramSze-1).get(new Integer(c+1)).intValue();
        }
        if(N_C_plus_1==0 || c>k){//i.e. this is the most frequent n-gram or the n-gram counts is higher than
            // the maximum allowed smoothing frequency as stated in Jurafsky and Martin Book.
            // then use MLE
            return(c);			
        }else{//i.e. use GT- smoothing formula
            if(c==0){
                return (double)((c+1)*N_C_plus_1)/(double)(N_C);
            }else{
                double N_K_plus_1=1;
                if(histogram.get(nGramSze-1).containsKey(new Integer(k+1))){// I used -1 because the unigram is stored on freq[0], bi-gram is stored in freq[1], ...etc
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
    
    /** Returns the backoff weight of a sequence of the input start and end indices
     * [Theta(end index| start index, ...,  end index-1) ].
     *
     * @return backoff weight in log scale to base (10) or 0 if the input ngram is not found
     */
    private double getTheta(String nGramStr) {
        int nGramSze=nGramStr.split("\\s+").length;
        if(freq.get(nGramSze-1).containsKey(nGramStr)){// I used -1 because the unigram is stored on freq[0], bi-gram is stored in freq[1], ...etc
            return 0.0;
        }else{
            String backOffStr=nGramStr.replaceAll(" [^ ]+$", "").trim();// remove the last word
            return getBackOff(backOffStr);
        }
    }
    
    /** Returns the backoff weight of a sequence of the input nGramStr
     *
     */
    private double getBackOff(String nGramStr) {
        if(alphas.containsKey(nGramStr)){
            return alphas.get(nGramStr);
        }else{
            return alphas.get(unk);
        }
    }
	
	/**
	 * @param args
	 * @throws IOException 
	 * @throws ClassNotFoundException 
	 */
	public static void main(String[] args) throws ClassNotFoundException, IOException {
		String LMPath=args[0];
		String testFilePath=args[1];

		Perplexity pp= new Perplexity(LMPath);
		double logPerplexity=pp.calculateLogPerplexity(testFilePath);
		System.out.println("Perplexity = "+logPerplexity);
		
		
		

	}

}
