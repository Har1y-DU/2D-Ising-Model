# 2D-Ising-Model
This code is for simulating the 2D Ising Model at different temperatures, and verifying the critical temperature T<sub>c</sub>, using Julia language. 

When excluding the external magnetic field, the Hamiltonian of Ising model can be written as:

![image](https://github.com/user-attachments/assets/bf24387e-68f1-4c77-9781-afab6c0761d1)
- J: exchange interaction strength, J > 0
- S: Spin of either +1 or -1

# Monte-Carlo Simulation
This code is using the Metropolis algorithm of Monte Carlo method to sweep the lattice and relax the system. The process is:
1. Create a random configuration of spin on a **N*N** lattice
2. Choose a random spin and flip its direction
3. Caluclate the change in energy of the system, **dE**
4. If **dE** < 0, then accept the flip; If **dE** < 0, then the move should be accepted with a probability of **exp<sup>-dE / T</sup>**
5. Repeat step 2-4, until the system reaches equilibrium

# Results
The value of **Energy**, **Magnetisation**, **Specific Heat**, and **Suscewptibility** are calculated, and plotted against different values of **Temperature**. The result using different lattice size **N** were plotted together to compare. 

![2D Ising Comparison](https://github.com/user-attachments/assets/89937bbe-f401-415d-b57e-d742eb95bb68)



