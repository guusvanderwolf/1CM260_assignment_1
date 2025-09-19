# -*- coding: utf-8 -*-
"""
@author: Guus van der Wolf
"""

import os
import numpy as np
import pandas as pd
import random
import math   


class Point2D: 
    """Class for representing a point in 2D space"""
    def __init__(self,id,x,y):
        self.id = id
        self.x = x
        self.y = y
    
    #method that computes the rounded euclidian distance between two 2D points
    def getDistance(c1,c2): 
        dx = c1.x-c2.x
        dy = c1.y-c2.y
        return math.sqrt(dx**2+dy**2)

class TSP:
    """
    Class for representing a Traveling Salesman Problem
    
    Attributes
    ----------
    nCities : int
        the number of cities
    cities : list of ints
        the cities, all represented by integers
    distMatrix : 2D array
        matrix with all distances between cities. Distance between city i and city j is distMatrix[i-1][j]
    
    """
    def __init__(self,tspFileName):
        """
        Reads a .tsp file and constructs an instance. 
        We assume that it is an Euclidian TSP

        Parameters
        ----------
        tspFileName : str
            name of the file
        """
        points = list() #add all points to list
        f = open(tspFileName)
        for line in f.readlines()[6:-1]: #start reading from line 7, skip last line
            asList = line.split()
            floatList = list(map(float,asList))

            id = int(floatList[0])-1 #convert to int, subtract 1 because Python indices start from 0
            x = floatList[1]
            y = floatList[2]

            c = Point2D(id,x,y)
            points.append(c)
        f.close()
        
        print("Read in all points, start computing distance matrix")

        
        self.nCities = len(points)
        self.cities = list(range(self.nCities))
        
        #compute distance matrix, assume Euclidian TSP
        self.distMatrix = np.zeros((self.nCities,self.nCities)) #init as nxn matrix
        for i in range(self.nCities):
            for j in range(i+1,self.nCities):
                distItoJ = Point2D.getDistance(points[i], points[j])
                self.distMatrix[i,j] = distItoJ
                self.distMatrix[j,i] = distItoJ
        
        print("Finished computing distance matrix")


    def getTour_NN(self,start):
        """
        Performs the nearest neighbour algorithm

        Parameters
        ----------
        start : int
            starting point of the tour

        Returns
        -------
        tour : list of ints
            order in which the cities are visitied.

        """
        tour = [start]
        notInTour = self.cities.copy()
        notInTour.remove(start)
    
        print("Start computing NN tour")
        for i in range(self.nCities-1):
            curCity = tour[i]
            closestDist = -1 #initialize with -1
            closestCity = None #initialize with None

            #find closest city not yet in tour
            for j in notInTour:
                dist = self.distMatrix[curCity][j]
                if dist<closestDist or closestCity is None:
                    #update the closest city and distance
                    closestDist = dist
                    closestCity = j
            
            tour.append(closestCity)
            notInTour.remove(closestCity)
        
        print("Finished computing NN tour")

        return tour

    def getTour_OutlierInsertion(self, start):
        """
        Construct a tour using the Outlier Insertion heuristic. Starts from an initial city and iteratively inserts the candidate city that is "furthest" from the current partial tour (the outlier). At each step, the city is inserted at the position in the tour that yields the smallest increase in cost

        Parameters
        ----------
        start : int
            starting point of the tour

        Returns
        -------
        tour : list of ints
            order in which the cities are visitied
        """
        tour = [start]

        #create the set of cities that are not yet in the tour (all except the start city)
        notInTour = self.cities.copy()
        notInTour.remove(start)

        print("Start computing Outlier Insertion tour")

        #add the city furthest from the start city to the tour
        first = self.furthest_city(notInTour, [start])
        tour.append(first)
        notInTour.remove(first)

        #add the outlier city iteratively to the tour
        while notInTour:
            candidate = self.furthest_city(notInTour, tour)
            pos = self.best_insertion_position(tour, candidate) #find best insertion position
            tour.insert(pos, candidate)
            notInTour.remove(candidate)

        print("Finished computing Outlier Insertion tour")
        return tour

    def getTour_GRASPedInsertion(self, start, alpha_pct=10.0, seed=None):
        """
        GRASP version of Outlier Insertion that randomizes the outlier selection step using an α% rule. 

        Parameters
        ----------
        start : int
            starting point of the tour
        alpha_pct : float
            α in percent
        seed : int
            RNG seed for reproducibility

        Returns
        -------
        tour : list of ints
            order in which the cities are visitied
        """
        rng = random.Random(seed) #initialize RNG
        tour = [start]

        #create the set of cities that are not yet in the tour (all except the start city)
        notInTour = self.cities.copy()
        notInTour.remove(start)

        print(f"Start computing GRASP Outlier Insertion tour (alpha={alpha_pct}%)")

        #add the city furthest from the start city to the tour
        first = self.furthest_city(notInTour, [start])
        tour.append(first)
        notInTour.remove(first)

        #add the outlier city iteratively to the tour
        while notInTour:
            scores = [(c, self.nearest_dist_to_tour(c, tour)) for c in notInTour] #compute outlier score for each candidate, which is the distance to the nearest city in current tour
            scores.sort(key=lambda x: x[1], reverse=True) #sort candidates descending by score (highest outlier score first)
            rcl_size = max(1, int(len(scores) * alpha_pct / 100)) #determine size of the RCL (at least 1 candidate)
            candidate = rng.choice(scores[:rcl_size])[0] #pick one candidate at random from the RCL
            pos = self.best_insertion_position(tour, candidate) #find best insertion position
            tour.insert(pos, candidate)
            notInTour.remove(candidate)

        print("Finished computing GRASP Outlier Insertion tour")
        return tour
    
    def isTwoOpt(self, tour, eps=1e-12):
        """
        Checks whether a tour is 2-optimal. A tour is 2-optimal if no pair of edges can be replaced by two other edges such that the total tour cost is reduced. For large instances (DIMENSION > 1379), the second arc is restricted to candidates where one endpoint is among the 40 nearest neighbors of either endpoint of the first arc

        Parameters
        ----------
        tour : list of ints
            current order in which the cities are visitied
        eps : float
            tolerance value, where small numerical gains below eps are ignored

        Returns
        -------
        bool
            True if the tour is 2-optimal, False otherwise
        """
        #check if the tours has less than 4 cities, becasue these are trivially 2-optimal
        n = len(tour)
        if n < 4:
            return True

        #consider the 40 nearest neighbors for large instances
        large = (self.nCities > 1379)
        if large:
            self.ensure_nn_lists(k=40)
            pos = {city: idx for idx, city in enumerate(tour)} #build a position map for O(1) lookup of city positions in the tour

        #try each edge (i, i+1) in the tour
        for i in range(n - 1):
            if large:
                a, b = tour[i], tour[(i + 1) % n] #ensures that when i is the last index, it loops back to 0
                cand_js = set()

                #build candidate j indices from neighbors of a and b
                for v in self.nn_lists[a] + self.nn_lists[b]:
                    j = pos.get(v)
                    if j is None:
                        continue
                    cand_js.add(j)
                    if j - 1 >= 0:
                        cand_js.add(j - 1)
                
                #check each candidate edge (j, j+1)
                for j in sorted(cand_js):
                    if j <= i + 1: #ensures edges do not overlap
                        continue
                    if i == 0 and j == n - 1: #skip case where we would break the closing edge of the tour
                        continue
                    if self.two_opt_gain(tour, i, j) > eps:
                        return False #found an improving 2-opt move
            else:
                #full neighborhood, check all pairs (i, j)
                for j in range(i + 2, n): 
                    if i == 0 and j == n - 1: #skip case where we would break the closing edge of the tour
                        continue
                    if self.two_opt_gain(tour, i, j) > eps:
                        return False #found an improving 2-opt move
        return True #found no improving 2-opt move

    def makeTwoOpt(self, tour, first_improvement=True, eps=1e-12):
        """
        Applies 2-opt local search to a given tour until it is 2-optimal. For large instances (DIMENSION > 1379), the second arc is restricted to candidates where one endpoint is among the 40 nearest neighbors of either endpoint of the first arc.

        Parameters
        ----------
        tour : list of ints
            current order in which the cities are visitied
        first_improvement : bool
            - if True, stop after the first improving move per iteration
            - if False, apply the best possible move in each iteration
        eps : float
            tolerance value, where small numerical gains below eps are ignored

        Returns
        -------
        tour : list of ints
            order in which the cities are visitied
        """
        #check if the tours has less than 4 cities, becasue these are trivially 2-optimal
        n = len(tour)
        if n < 4:
            return tour

        #consider the 40 nearest neighbors for large instances
        large = (self.nCities > 1379)
        if large:
            self.ensure_nn_lists(k=40)
            pos = {city: idx for idx, city in enumerate(tour)} #build a position map for O(1) lookup of city positions in the tour

        #set initial values
        improved = True
        while improved:
            improved = False
            best_gain, best_i, best_j = 0.0, None, None

            #try each edge (i, i+1) in the tour
            for i in range(n - 1):
                if large:
                    a, b = tour[i], tour[(i + 1) % n] #ensures that when i is the last index, it loops back to 0
                    cand_js = set()

                    #build candidate j indices from neighbors of a and b
                    for v in self.nn_lists[a] + self.nn_lists[b]:
                        j = pos.get(v)
                        if j is None:
                            continue
                        cand_js.add(j)
                        if j - 1 >= 0:
                            cand_js.add(j - 1)

                    #check each candidate edge (j, j+1)
                    for j in sorted(cand_js):
                        if j <= i + 1:  #ensures edges do not overlap
                            continue
                        if i == 0 and j == n - 1:  #skip case where we would break the closing edge of the tour
                            continue

                        gain = self.two_opt_gain(tour, i, j)
                        if gain > eps:
                            if first_improvement:
                                tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1]) #apply the reverse move
                                improved = True
                                break

                            #track the best move to apply later
                            elif gain > best_gain:
                                best_gain, best_i, best_j = gain, i, j

                else:
                    #full neighborhood, check all pairs (i, j)
                    for j in range(i + 2, n):
                        if i == 0 and j == n - 1:  #skip case where we would break the closing edge of the tour
                            continue
                        gain = self.two_opt_gain(tour, i, j)
                        if gain > eps:
                            if first_improvement:
                                tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                                improved = True
                                break
                            elif gain > best_gain:
                                best_gain, best_i, best_j = gain, i, j

                if improved and first_improvement:
                    break #restart search after applying a move

            #apply the best move found in this iteration
            if not first_improvement and best_gain > eps:
                tour[best_i + 1:best_j + 1] = reversed(tour[best_i + 1:best_j + 1])
                improved = True

        return tour
    
    def getTour_GRASP2Opt(self, start, alpha_pct = 10.0, seed=None):
        """
        Construct a tour with GRASPed Outlier Insertion and improve it with 2-opt.

        Parameters
        ----------
        start : int
            starting point of the tour
        alpha_pct : float
            α in percent (default 10.0)
        seed : int
            RNG seed for reproducibility

        Returns
        -------
        tour_after : list of ints
            order in which the cities are visitied
        """
        #run the GRASPed Outlier Insertion heuristic followed by the 2-opt heuristic
        tour_before = self.getTour_GRASPedInsertion(start=start, alpha_pct=alpha_pct, seed=seed)
        tour_after = self.makeTwoOpt(tour_before)
        return tour_after
    
    def best_insertion_position(self, tour, city):
        """
        Find the best position in a tour to insert a new city. The position is chosen such that the increase in tour cost is minimized

        Parameters
        ----------
        tour : list of ints
            current partial tour
        city : int
            index of the city to insert into the tour

        Returns
        -------
        best_pos : tuple of (int, float)
            the best insertion position and the corresponding increase in cost
        """
        best_pos = None
        best_increase = float("inf")

        #try to insert the city between every consecutive pair of cities (i, j)
        for i in range(len(tour)):
            j = (i + 1) % len(tour) #ensure last city is paired with the first city to keep the tour circular

            #compute the added cost of breaking the edge (i, j) and replacing it with the edges (i, city) and (city, j)
            increase = (self.distMatrix[tour[i]][city] + self.distMatrix[city][tour[j]] - self.distMatrix[tour[i]][tour[j]])

            #keep track of the insertion position with the lowest added cost
            if increase < best_increase:
                best_increase = increase
                best_pos = j

        return best_pos
    
    def nearest_dist_to_tour(self, city, tour):
        """
        Compute the distance from a city to its nearest neighbor in a given tour

        Parameters
        ----------
        city : int
            index of the city for which we want the distance
        tour : list of ints
            current order in which the cities are visitied

        Returns
        -------
        float
            distance from the city to the closest city in tour
        """
        best = None

        #check the distance from the city to every city in the current tour
        for i in tour:
            d = self.distMatrix[city][i]
            #update best if this is the first distance computed or if we found a shorter distance
            if best is None or d < best: 
                best = d
        return best

    def furthest_city(self, candidates, from_set):
        """
        Find the city among the candidates that is furthest from the given set

        Parameters
        ----------
        candidates : list of ints
            cities that can be chosen from
        from_set : list of ints
            set of cities already in the tour

        Returns
        -------
        furhestCity : int
            the city from candidates that is the furthest from from_set
        """
        #set initial values
        furthestCity = None
        furthestDist = -1.0

        for c in candidates:
            #compute distance from c to its nearest neighbor in from_set
            nearest = None
            for i in from_set:
                d = self.distMatrix[c][i]
                if nearest is None or d < nearest:
                    nearest = d
            #update if this candidate is further than previous best        
            if nearest > furthestDist:
                furthestDist = nearest
                furthestCity = c

        return furthestCity
    
    def two_opt_gain(self, tour, i, j):
        """
        Computes the change in tour length for a 2-opt move that reverses the segment (i+1 .. j)

        Parameters
        ----------
        tour : list of ints
            current order in which the cities are visitied
        i : int
            index of the first edge (i, i+1) in the tour
        j : int
            index of the second edge (j, j+1) in the tour

        Returns
        -------
        gain: float
            gain = removed length − added length. 
        """
        n = len(tour)

        #find the endpoints of the two edges to break (a,b) and (c,d)
        a = tour[i]
        b = tour[(i + 1) % n]  #ensures the tour is circular
        c = tour[j]
        d = tour[(j + 1) % n]

        removed = self.distMatrix[a][b] + self.distMatrix[c][d] #compute the total length of edges removed by the 2-opt swap
        added = self.distMatrix[a][c] + self.distMatrix[b][d] #compute the total length of edges added after reconnecting

        return removed - added

    def ensure_nn_lists(self, k=40):
        """
        Precomputes (and caches) the k nearest neighbors per city (excluding the city itself)

        Parameters
        ----------
        k : int
            number of nearest neighbors to keep for each city (default 40)

        Returns
        -------
        self.nn_lists : list[list[int]]
            neighbor lists which contains the k closest cities to city u, ordered from closest to farthest
        """
        #reuse the cached neighbor lists if they exist and have size k
        if getattr(self, "nn_lists", None) is not None and len(self.nn_lists[0]) == k:
            return self.nn_lists
        
        n = self.nCities
        self.nn_lists = []
        for i in range(n):
            order = np.argsort(self.distMatrix[i]) #sort all cities by distance from city i (ascending order)
            order = [j for j in order if j != i][:k] #remove i, then keep only the first k closest cities
            self.nn_lists.append(order) #store the k nearest neighbors for city i
    
    def getCitiesCopy(self): 
        return self.cities.copy()
        
    def evaluateSolution(self,tour):
        if self.isFeasible(tour):
            costs = self.computeCosts(tour)
            print("The solution is feasible with costs "+str(costs))
        else: 
            print("The solution is infeasible")
    
    def isFeasible(self,tour):
        """
        Checks if tour is feasible

        Parameters
        ----------
        tour : list of integers
            order in which cities are visited. For a 4-city TSP, an example tour is [3, 1, 4, 2]

        Returns
        -------
        bool
            TRUE if feasible, FALSE if infeasible.

        """
        #first check if the length of the tour is correct
        if len(tour)!=self.nCities:
            print("Length of tour incorrect")
            return False
        else: 
            #check if all cities in the tour
            for city in self.cities:
                if city not in tour:
                    return False
        return True
    
    def computeCosts(self,tour):
        """
        Computes the costs of a tour

        Parameters
        ----------
        tour : list of integers
            order of cities.

        Returns
        -------
        costs : int
            costs of tour.

        """
        costs = 0
        for i in range(len(tour)-1):
            costs += self.distMatrix[tour[i],tour[i+1]]
            
        #add the costs to complete the tour back to the start
        costs += self.distMatrix[tour[-1],tour[0]]
        return costs
    

