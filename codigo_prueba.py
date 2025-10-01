#6. Función acumuladora: Promedio hasta que se escriba "fin" 
#Instrucciones: 
#Escribí una función promediar() que: 
#● Pida al usuario que ingrese números uno por uno. 
#● Termine cuando el usuario escriba "fin". 
#● Devuelva el promedio de los valores ingresados.

def promediar():

    contador = 0
    acumulador = 0

    while True:
      entrada = input("ingrese los numeros que desea promediar o escriba 'fin' para finalizar ")

      if entrada == "fin":
         break
      
      else:
            numero = float(entrada)  # convierto a número (acepta decimales)
            acumulador += numero
            contador += 1
        #except ValueError:
            #print("Eso no es un número válido. Intenta de nuevo.")
    
    if acumulador > 0:
        return acumulador / contador
    else:
        return 0   # si no se ingresó ningún número

######

# Uso
print("El promedio es:", promediar())

# Hola Dei, hice un cambio 
