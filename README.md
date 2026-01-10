# Projeto29
<p align="justify">
  



  SMOTE-Bagging é um algoritmo que não está no sci-kit learn.
  
  O que o algoritmo faz?
  
  O script implementa uma técnica chamada SMOTE-Bagging:
  
      Paralelismo: Ele cria 6 versões diferentes do dataset usando SMOTE (com sementes aleatórias diferentes).
  
      Independência: Ele treina um XGBClassifier em cada uma dessas versões.
  
      Votação (Averaging): No final, ele tira a média das predições para decidir o resultado.
      
      
      
  O SMOTE-Bagging se mostrou superior a outros algoritmos para dados desbalanceados, existentes no sci-kit learn

</p>
