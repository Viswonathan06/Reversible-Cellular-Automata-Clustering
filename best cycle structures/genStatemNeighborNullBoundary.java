package sufficientCellLength;

import java.io.IOException;
import java.util.Scanner;

import javax.swing.JOptionPane;

public class genStatemNeighborNullBoundary {
	

	public static void main(String[] args)
	{
		int d, n,m,left,right;
		int	i, index;   
		d = Integer.parseInt(JOptionPane.showInputDialog("Enter the state of the CA"));
		//d=3;
		left = Integer.parseInt(JOptionPane.showInputDialog("Enter the number of left neighbors"));
		right = Integer.parseInt(JOptionPane.showInputDialog("Enter the number of right neighbors"));
		
		m=left+right+1; //m is the number of neighboring cells
		
		n = Integer.parseInt(JOptionPane.showInputDialog("Enter the number of cells"));
		//n=6;
		
		int N = (int)Math.pow(d, n);	/* N = total no. of states of an n-cell CA */
		
		int noRMTs = (int)Math.pow(d, m);
		int Rule[]= new int [noRMTs];
		String ruleString = new String();
		ruleString = JOptionPane.showInputDialog("Enter the rule for "+d+" state " + m +" neighborhood CA :");
		//ruleString ="00011110"; //rule 30
		
		
		int j=0;
		for(i = noRMTs-1; i>=0; i--){
			Rule[j++] = Integer.parseInt(String.valueOf(ruleString.charAt(i)));
		}	
		System.out.println("The rule of the "+d+" state CA is");
		for(i =noRMTs-1; i>=0; i--){
			System.out.print(Rule[i]);//Rule[i]=input.nextInt();
		}
		System.out.println();
		
		int check[] = new int[N];
		int PS[] = new int[n];
		int SS[] = new int[n];
		int NS[] = new int[n];
		int Comb[] = new int[n];
		
		Boolean flag = false;
		for(i=0;i<N;i++)
			check[i] = 0;
		
		for(i=0;i<n;i++)
			PS[i] = Comb[i] = 0;
		
		for(i=0;i<n;i++)
			System.out.print(PS[i]);
		System.out.println(" (0)");
		check[0] = 1;
		while(true){
			for(i=0;i<n;i++)
				SS[i] = PS[i];
			while(true)
			{		
				/* Next state generation and cycle checking */
				for(i=0;i<left;i++){  //correct for leftmost cells
					//calculation of RMT
					int RMT =0, range = m-left-1+i;
					
					for(j = 0;j<m-left+i;j++){
						RMT += (int)( Math.pow(d, range)*PS[j]);
						range--;
					}
					NS[i] = Rule[RMT];  // next state for ith cell
				}
				for(i=left;i<n-right;i++){
					//calculation of RMT
					int RMT =0, range = m-1;
					
					for(j = i-left;j<=i+right;j++){
						RMT += (int)( Math.pow(d, range)*PS[j]);
						range--;
					}
					NS[i] = Rule[RMT];  // next state for ith cell
				}
				for(i=n-right;i<n;i++){ //correct for rightmost cells
					//calculation of RMT
					int RMT =0, range = m-1;
					
					for(j = i-left;j<n;j++){
						RMT += (int)( Math.pow(d, range)*PS[j]);
						range--;
					}
					NS[i] = Rule[RMT];  // next state for ith cell
				}
				
				for(i=0;i<n;i++)
					PS[i] = NS[i];
				for(i=0;i<n;i++)
					System.out.print(PS[i]);
			
				index = 0;
				for(i=0;i<n;i++)
					index += PS[i]*Math.pow(d,n-i-1);
				System.out.println(" (" +index +") ");
	
				if(check[index]==1){
					System.out.println();
					break;	
				}
				else check[index] = 1;
			}
			for(i=0;i<n && SS[i]==PS[i];i++);
			/*if(i!=n)
			{
				System.out.println("\nIrreversible for " + n + "\n");
				break;
			}*/
			
			while(true)
			{	/* Find an unexplored state */
				for(i=0;i<n;i++)
					if(Comb[i]==(d-1))
						Comb[i] = 0;
					else	break;
				if(i<n)
				{
					Comb[i] += 1;
				}
				else	break;
	
				index = 0;
				for(i=0;i<n;i++)
					index += Comb[i]*Math.pow(d,n-i-1);
				
				if(check[index]==0)
				{
					
					for(i=0;i<n;i++)
						System.out.print(Comb[i]);
					System.out.println(" (" +index +")");
					
					check[index] = 1;
					break;
				}
			}
			for(i=0;i<n && Comb[i]==0;i++);
			if(i==n)
			{
				flag = true;
				break;
			}
			for(i=0;i<n;i++)
				PS[i] = Comb[i];
			
		}
		
	/*	if(flag==true)
		{
			System.out.print("\nReversible for #cells = "+n);
			System.out.println();
		}
		else{
			System.out.print("\nIrreversible for #cells = "+n);
			System.out.println();
		}
			*/
	}

}
