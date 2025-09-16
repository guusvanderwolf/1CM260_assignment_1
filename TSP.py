# -*- coding: utf-8 -*-
"""
@author: Guus van der Wolf
"""

import numpy as np
import math   
import os


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

    # ========= deterministic Outlier Insertion =========
    def getTour_OutlierInsertion(self, start):
        """
        Construct a tour using the Outlier Insertion heuristic.

        The method starts from an initial city and iteratively inserts the
        candidate city that is "furthest" from the current partial tour
        (the outlier). At each step, the city is inserted at the position
        in the tour that yields the smallest increase in cost.

        Parameters
        ----------
        start : int
            Index of the starting city.

        Returns
        -------
        list of int
            A complete tour (list of city indices), beginning with `start`.
        """
        tour = [start]
        notInTour = self.cities.copy()
        notInTour.remove(start)

        print("Start computing Outlier Insertion tour")

        # first extension: furthest from start
        first = self.furthest_city(notInTour, [start])
        tour.append(first)
        notInTour.remove(first)

        # iterative insertions
        while notInTour:
            candidate = self.furthest_city(notInTour, tour)
            pos, _ = self.best_insertion_position(tour, candidate)
            tour.insert(pos, candidate)
            notInTour.remove(candidate)

        print("Finished computing Outlier Insertion tour")
        return tour

    # ========= GRASP'ed Outlier Insertion (randomized both steps) =========
    def getTour_GRASPedInsertion(self, start, alpha_pct=10.0, seed=None):
        """
        GRASP version of Outlier Insertion that randomizes *both* steps using an α% rule.

        Parameters
        ----------
        start : int
            starting city
        alpha_pct : float
            α in percent (e.g., 10 means 'within 10% of the best'):
            - City selection (maximize): choose uniformly at random among cities whose
              'distance to nearest tour city' ≥ (1-α) * bestScore.
            - Position selection (minimize): choose uniformly among positions whose
              insertion increase ≤ (1+α) * bestIncrease.
        seed : int | None
            RNG seed for reproducibility.

        Returns
        -------
        tour : list[int]
        """
        import random
        rng = random.Random(seed)
        alpha = float(alpha_pct) / 100.0

        tour = [start]
        notInTour = self.cities.copy()
        notInTour.remove(start)

        print("Start computing GRASPed Outlier Insertion tour")

        # ---- first extension: furthest from start with α-RCL ----
        # scores = distance to start
        scores = {c: self.distMatrix[start][c] for c in notInTour}
        bestScore = max(scores.values())
        thr_city = (1.0 - alpha) * bestScore
        rcl_cities = [c for c, s in scores.items() if s >= thr_city]
        first = rng.choice(rcl_cities)
        tour.append(first)
        notInTour.remove(first)

        # ---- iterative insertions: randomize city & position with α-RCLs ----
        while notInTour:
            # CITY RCL (maximize nearest-to-tour distance)
            scores = {c: self.nearest_dist_to_tour(c, tour) for c in notInTour}
            bestScore = max(scores.values())
            thr_city = (1.0 - alpha) * bestScore
            rcl_city = [c for c, s in scores.items() if s >= thr_city]
            city = rng.choice(rcl_city)

            # POSITION RCL (minimize insertion increase)
            # gather all positions and best increase
            m = len(tour)
            opts = []
            bestInc = None
            for k in range(m):
                i, j = tour[k], tour[(k + 1) % m]
                inc = (self.distMatrix[i][city]
                       + self.distMatrix[city][j]
                       - self.distMatrix[i][j])
                opts.append((k + 1, inc))
                if bestInc is None or inc < bestInc:
                    bestInc = inc
            thr_pos = (1.0 + alpha) * bestInc
            rcl_pos = [pos for (pos, inc) in opts if inc <= thr_pos]
            pos = rng.choice(rcl_pos)

            tour.insert(pos, city)
            notInTour.remove(city)

        print("Finished computing GRASPed Outlier Insertion tour")
        return tour
    
    def isTwoOpt(self, tour, eps=1e-12):
        """
        Checks whether a tour is 2-optimal.

        A tour is 2-optimal if no pair of edges can be replaced by two other edges
        such that the total tour cost is reduced.

        For large instances (DIMENSION > 1379), the second arc is restricted to
        candidates where one endpoint is among the 40 nearest neighbors of either
        endpoint of the first arc. This speeds up the check significantly.

        Parameters
        ----------
        tour : list of int
            Order in which cities are visited. Example: [0, 2, 1, 3].
        eps : float, optional
            Tolerance value. Small numerical gains below eps are ignored.

        Returns
        -------
        bool
            True if the tour is 2-optimal (cannot be improved by any 2-exchange),
            False otherwise.
        """
        n = len(tour)
        if n < 4:
            return True

        large = (self.nCities > 1379) # For large instances, only consider the 40 nearest neighbors
        if large:
            self._ensure_nn_lists(k=40)
            # position map for O(1) index lookup
            pos = {city: idx for idx, city in enumerate(tour)}

        for i in range(n - 1):
            if large:
                a, b = tour[i], tour[(i + 1) % n]
                cand_js = set()

                # if city v is a neighbor of a or b, then an eligible second edge
                # is the edge starting at index j where v == tour[j] (or j-1).
                for v in self._nn_lists[a] + self._nn_lists[b]:
                    j = pos.get(v)
                    if j is None:
                        continue
                    cand_js.add(j)
                    if j - 1 >= 0:
                        cand_js.add(j - 1)
                # filter and check candidates
                for j in sorted(cand_js):
                    if j <= i + 1:
                        continue
                    if i == 0 and j == n - 1:  # skip closing edge as pair
                        continue
                    if self._two_opt_gain(tour, i, j) > eps:
                        return False
            else:
                # full neighborhood
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1: # Skip case where we would "break" the closing edge of the tour
                        continue
                    if self._two_opt_gain(tour, i, j) > eps:
                        return False
        return True

    def makeTwoOpt(self, tour, first_improvement=True, eps=1e-12):
        """
        Applies 2-opt local search to a given tour until it is 2-optimal.

        The algorithm repeatedly checks whether replacing two edges with two
        different edges reduces the cost. If so, the segment between the edges
        is reversed. The process stops once no further improvement is possible.

        For large instances (DIMENSION > 1379), the second arc is restricted to
        candidates where one endpoint is among the 40 nearest neighbors of either
        endpoint of the first arc.

        Parameters
        ----------
        tour : list of int
            Initial tour to be improved. Example: [0, 2, 1, 3].
        first_improvement : bool, optional
            If True, stop after the first improving move per iteration.
            If False, apply the best possible move in each iteration.
        eps : float, optional
            Tolerance value. Small numerical gains below eps are ignored.

        Returns
        -------
        list of int
            The improved tour that is 2-optimal under the given neighborhood definition.
        """
        n = len(tour)
        if n < 4:
            return tour

        large = (self.nCities > 1379)
        if large:
            self._ensure_nn_lists(k=40)

        improved = True
        while improved:
            improved = False
            # rebuild position map at the start of each pass (tour may change)
            pos = {city: idx for idx, city in enumerate(tour)}

            for i in range(n - 1):
                if large:
                    a, b = tour[i], tour[(i + 1) % n]
                    cand_js = set()
                    for v in self._nn_lists[a] + self._nn_lists[b]:
                        j = pos.get(v)
                        if j is None:
                            continue
                        cand_js.add(j)
                        if j - 1 >= 0:
                            cand_js.add(j - 1)
                    iter_js = sorted(cand_js)
                else:
                    iter_js = range(i + 2, n)

                for j in iter_js:
                    if j <= i + 1:
                        continue
                    if i == 0 and j == n - 1:
                        continue
                    gain = self._two_opt_gain(tour, i, j)
                    if gain > eps:
                        # accept: reverse segment (i+1 .. j)
                        tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1]) # Reverse the segment between i+1 and j (inclusive)
                        improved = True
                        if first_improvement:
                            break
                if improved and first_improvement:
                    break
        return tour
    
    def getTour_GRASP2Opt(self, start: int, alpha_pct: float = 10.0, seed: int | None = None):
        """
        Construct a tour with GRASPed Outlier Insertion and improve it with 2-opt.

        Parameters
        ----------
        start : int
            Starting city.
        alpha_pct : float, optional
            α percentage for GRASP restricted candidate lists (default 10.0).
        seed : int | None, optional
            RNG seed forwarded to the GRASP builder.

        Returns
        -------
        list[int]
            2-opt–improved tour.
        """
        tour_before = self.getTour_GRASPedInsertion(start=start, alpha_pct=alpha_pct, seed=seed)
        tour_after = self.makeTwoOpt(tour_before)
        return tour_after
    
    def best_insertion_position(self, tour, city):
        """
        Find the best position in a tour to insert a new city.

        The position is chosen such that the increase in tour cost is minimized.
        The method evaluates all possible insertion edges (i, i+1) in the
        current tour, computes the cost of inserting the city between them,
        and selects the best option.

        Parameters
        ----------
        tour : list of int
            Current partial tour (sequence of cities).
        city : int
            Index of the city to insert into the tour.

        Returns
        -------
        tuple of (int, float)
            The best insertion position (index where the city should be inserted)
            and the corresponding increase in cost.
        """
        bestPos = 0
        bestIncrease = None
        m = len(tour)
        for k in range(m):
            i = tour[k]
            j = tour[(k + 1) % m]
            increase = (self.distMatrix[i][city]
                        + self.distMatrix[city][j]
                        - self.distMatrix[i][j])
            if bestIncrease is None or increase < bestIncrease:
                bestIncrease = increase
                bestPos = k + 1
        return bestPos, bestIncrease
    
    def nearest_dist_to_tour(self, city, tour):
        """
        Compute the distance from a city to its nearest neighbor in a given tour.

        Parameters
        ----------
        city : int
            Index of the city for which we want the distance.
        tour : list of int
            Sequence of cities forming the current tour.

        Returns
        -------
        float
            Distance from `city` to the closest city in `tour`.
        """
        best = None
        for i in tour:
            d = self.distMatrix[city][i]
            if best is None or d < best:
                best = d
        return best

    def furthest_city(self, candidates, from_set):
        """
        Find the city among the candidates that is furthest from the given set.

        The "distance" of a candidate city is defined as its distance to the
        *nearest* city in `from_set`. The city maximizing this value is returned.

        Parameters
        ----------
        candidates : list of int
            Cities that can be chosen from.
        from_set : list of int
            Set of cities already in the tour.

        Returns
        -------
        int
            The city in `candidates` with the largest minimum distance to `from_set`.
        """
        furthestCity = None
        furthestDist = -1.0
        for c in candidates:
            # distance to *nearest* city already in from_set
            nearest = None
            for i in from_set:
                d = self.distMatrix[c][i]
                if nearest is None or d < nearest:
                    nearest = d
            if nearest > furthestDist:
                furthestDist = nearest
                furthestCity = c
        return furthestCity
    
    def _two_opt_gain(self, tour, i, j):
        """
        Computes the change in tour length for a 2-opt move that reverses
        the segment (i+1 .. j).

        The move replaces edges (tour[i], tour[i+1]) and (tour[j], tour[j+1])
        by (tour[i], tour[j]) and (tour[i+1], tour[j+1]). The gain is computed
        in O(1) by touching only these four edges.

        Parameters
        ----------
        tour : list of int
            Current tour; tour[t] gives the city at position t (tour is circular).
        i : int
            Index of the first edge (i, i+1) in the tour.
        j : int
            Index of the second edge (j, j+1) in the tour, with j >= i + 2.

        Returns
        -------
        float
            Gain = (removed length) − (added length).
            A positive value means the swap improves the tour.
        """
        n = len(tour)
        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[j], tour[(j + 1) % n]
        removed = self.distMatrix[a][b] + self.distMatrix[c][d]
        added   = self.distMatrix[a][c] + self.distMatrix[b][d]
        return removed - added  # > 0 ⇒ improvement

    def _ensure_nn_lists(self, k=40):
        """
        Precomputes (and caches) k nearest neighbors per city (excluding the city itself).

        The result is stored in self._nn_lists so it can be reused by 2-opt routines.
        If the cache already exists with the requested k, it is reused.

        Parameters
        ----------
        k : int, optional
            Number of nearest neighbors to keep for each city (default 40).

        Returns
        -------
        list[list[int]]
            Neighbor lists; self._nn_lists[u] contains the k closest cities to city u,
            ordered from closest to farthest.
        """
        if getattr(self, "_nn_lists", None) is not None and len(self._nn_lists[0]) == k:
            return
        n = self.nCities
        self._nn_lists = []
        for i in range(n):
            order = np.argsort(self.distMatrix[i]) # Sort distances from city i (ascending)
            order = [j for j in order if j != i][:k] # Drop self and keep first k
            self._nn_lists.append(order)
    
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
            
        # add the costs to complete the tour back to the start
        costs += self.distMatrix[tour[-1],tour[0]]
        return costs
    

