#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esta classe deve conter todas as suas implementações relevantes para seu filtro de partículas
"""

from pf import Particle, create_particles, draw_random_sample
import numpy as np
import inspercles # necessário para o a função nb_lidar que simula o laser
import math
from scipy.stats import norm



largura = 775 # largura do mapa
altura = 748  # altura do mapa

# Robo
robot = Particle(largura/2, altura/2, math.pi/4, 1.0)

# Nuvem de particulas
particulas = []

num_particulas = 200


# Os angulos em que o robo simulado vai ter sensores
angles = np.linspace(0.0, 2*math.pi, num=8, endpoint=False)

# Lista mais longa
movimentos_longos = [[-10, -10, 0], [-10, 10, 0], [-10,0,0], [-10, 0, 0],
				[0,0,math.pi/12.0], [0, 0, math.pi/12.0], [0, 0, math.pi/12],[0,0,-math.pi/4],
				[-5, 0, 0],[-5,0,0], [-5,0,0], [-10,0,0],[-10,0,0], [-10,0,0],[-10,0,0],[-10,0,0],[-15,0,0],
				[0,0,-math.pi/4],[0, 10, 0], [0,10,0], [0, 10, 0], [0,10,0], [0,0,math.pi/8], [0,10,0], [0,10,0], 
				[0,10,0], [0,10,0], [0,10,0],[0,10,0],
				[0,0,-math.radians(90)],
				[math.cos(math.pi/3)*10, math.sin(math.pi/3),0],[math.cos(math.pi/3)*10, math.sin(math.pi/3),0],[math.cos(math.pi/3)*10, math.sin(math.pi/3),0],
				[math.cos(math.pi/3)*10, math.sin(math.pi/3),0]]

# Lista curta
movimentos_curtos = [[-10, -10, 0], [-10, 10, 0], [-10,0,0], [-10, 0, 0]]

movimentos_relativos = [[0, -math.pi/3],[10, 0],[10, 0], [10, 0], [10, 0],[15, 0],[15, 0],[15, 0],[0, -math.pi/2],[10, 0],
						[10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
						[10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
						[10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
						[0, -math.pi/2], 
						[10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
						[10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
						[10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
						[10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
						[0, -math.pi/2], 
						[10,0], [0, -math.pi/4], [10,0], [10,0], [10,0],
						[10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
						[10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0]]



movimentos = movimentos_relativos



def cria_particulas(minx=0, miny=0, maxx=largura, maxy=altura, n_particulas=num_particulas):
	"""
		Cria uma lista de partículas distribuídas de forma uniforme entre minx, miny, maxx e maxy
	"""
	particle_cloud = []
	for i in range(n_particulas):
		x = np.random.uniform(minx,maxx)
		y = np.random.uniform(miny,maxy)
		theta = np.random.uniform(0,2*math.pi)
		p = Particle(x, y, theta) # A prob. w vai ser normalizada depois
		particle_cloud.append(p)
	return particle_cloud
    
def move_particulas(particulas, movimento):
	"""
	Recebe um movimento na forma [deslocamento, theta]  e o aplica a todas as partículas
	Assumindo um desvio padrão para cada um dos valores
	Esta função não precisa devolver nada, e sim alterar as partículas recebidas.

	Sugestão: aplicar move_relative(movimento) a cada partícula

	Você não precisa mover o robô. O código fornecido pelos professores fará isso

	"""
	for parti in particulas:    
		mov = [movimento[0],movimento[1]]      
		mov[0] += np.random.normal(0,1.5,len(particulas))[0]
		mov[1] += np.random.normal(0,math.radians(1.5),len(particulas))[0]
		parti.move_relative(mov)
	return particulas


def leituras_laser_evidencias(robot, particulas):
	"""
	Realiza leituras simuladas do laser para o robo e as particulas
	Depois incorpora a evidência calculando
	P(H|D) para todas as particulas
	Lembre-se de que a formula $P(z_t | x_t) = \alpha \prod_{j}^M{e^{\frac{-(z_j - \hat{z_j})}{2\sigma^2}}}$ 
	responde somente P(Hi|D), em que H é a hi

	Esta função não precisa retornar nada, mas as partículas precisa ter o seu w recalculado. 

	Você vai precisar calcular para o robo

	"""
	# Voce vai precisar calcular a leitura para cada particula usando inspercles.nb_lidar e depois atualizar as probabilidades
	
	leitura_robo = inspercles.nb_lidar(robot, angles)
	lista_dis = []
	dis = 0
	total = 0

	for p in particulas:
		leitura_part = inspercles.nb_lidar(p,angles)
		val_som = 0
		for l in leitura_part:
			distancia_r = leitura_robo[l]
			distancia_p = leitura_part[l]
			dis = norm.pdf(distancia_p,loc = distancia_r, scale = 10)
			val_som += dis
		p.w = val_som
		total += val_som		
	for p in particulas:
		p.normalize(total)

def reamostrar(particulas, n_particulas = num_particulas):
	"""
	Reamostra as partículas devolvendo novas particulas sorteadas
	de acordo com a probabilidade e deslocadas de acordo com uma variação normal    

	O notebook como_sortear tem dicas que podem ser úteis

	Depois de reamostradas todas as partículas precisam novamente ser deixadas com probabilidade igual

	Use 1/n ou 1, não importa desde que seja a mesma
	"""
	pesos = []
	for p in particulas:
		pesos.append(p.w)

	particulas = draw_random_sample(particulas, pesos, n_particulas)

	desvio = 9

	for p in particulas:
		p.x += np.random.normal(0,desvio,1)[0]
		p.y += np.random.normal(0,desvio,1)[0]
		p.theta += np.random.normal(0,math.radians(15), 1)[0]
		p.w = 1
	return particulas
