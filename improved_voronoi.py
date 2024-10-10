"""
    File:        improved_voronoi.py
    Author:      Zoe Carovano
    Course:      CS 330 - Algorithms
    Semester:    Spring 2024
    Assignment:  Voronoi Diagram Improved Implementation:
                 Divide and Conquer Voronoi
"""

import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import math
import sympy as sympy
from sympy.abc import x, y
import numpy as np
import shapely.geometry

# CITE: https://shapely.readthedocs.io/en/stable/geometry.html
# DESC: Used for understanding shapely.geometry package

BOUNDING_BOX = []

class HalfEdge:

    def __init__(self, twin, center_point, edge_points):
        self.twin = twin
        self.center_point = center_point
        self.edge_points = edge_points     # Stored CW

    def get_twin(self):
        return self.twin
    
    def get_center_point(self):
        return self.center_point
    
    def get_edge_points(self):
        return self.edge_points
    
    def set_twin(self, new_twin):
        self.twin = new_twin
        return
    
    def set_edge_points(self, new_edge_points):
        self.edge_points = new_edge_points
        return
    
class VoronoiCell:

    def __init__(self, center_point, half_edges):
        self.center_point = center_point
        self.half_edges = half_edges       # List ordered CW

    def get_center_point(self):
        return self.center_point
    
    def get_half_edges(self):
        return self.half_edges
    
    def intersection(self, linear_bisector):
        line = []
        line_segment = []

        #See where bisector intersects bounding box.
        for i in range(len(BOUNDING_BOX)):
            if i % 2:
                intersection_point = BOUNDING_BOX[i][0], linear_bisector.evalf(3, subs={x: BOUNDING_BOX[i][0]})
                if BOUNDING_BOX[0][1] <= intersection_point[1] <= BOUNDING_BOX[1][1]:
                    line_segment.append(intersection_point)
                    line.append(intersection_point)

            else: 
                intersection_point = sympy.solve(sympy.Eq(linear_bisector, BOUNDING_BOX[i][1]), x)[0].evalf(3), BOUNDING_BOX[i][1]
                if BOUNDING_BOX[0][0] < intersection_point[0] < BOUNDING_BOX[2][0]:
                    line_segment.append(intersection_point)
                    line.append(intersection_point)
            
        bisector_line = shapely.LineString(line_segment)

        intersections = []

        for half_edge in self.half_edges:
            half_edge_points = half_edge.get_edge_points()
            region_line = shapely.LineString(half_edge_points)
            if region_line.intersects(bisector_line):
                intersection_point = region_line.intersection(bisector_line)
                intersections.append(Intersection((intersection_point.x, intersection_point.y), half_edge))

        return intersections
    
    def update_cell(self, new_half_edges):

        # Vertex where we start inserting new edges.
        start_replacing_point = new_half_edges[0].edge_points[0]

        # Vertex where we stop inserting new edges.
        done_replacing_point = new_half_edges[-1].edge_points[1]

        keeping = False
        started_replacing = False
        updated_half_edges = []

        #First loop: Set keeping to true when see before_edge.
        for half_edge in self.half_edges:
            # Before edge = start replacing point is the end point of current half edge
            # Should we start replacing yet?
            if half_edge.edge_points[1] == start_replacing_point:
                updated_half_edges.append(half_edge)
                # Append all new half edges to updated half edge list.
                for new_edge in new_half_edges:
                    updated_half_edges.append(new_edge)
                started_replacing = True
            # After edge = done replacing point is the start point of current half edge
            # Should we stop replacing?
            elif half_edge.edge_points[0] == done_replacing_point and started_replacing:
                updated_half_edges.append(half_edge)
                keeping = True
            elif keeping:
                updated_half_edges.append(half_edge)

        # If already found before edge but not after edge, loop through again.
        for half_edge in self.half_edges:
            if half_edge.edge_points[0] == done_replacing_point:
                updated_half_edges.append(half_edge)
                keeping = True
            elif half_edge.edge_points[1] == start_replacing_point:
                self.half_edges = updated_half_edges
                return
            elif keeping:
                updated_half_edges.append(half_edge)
    
    def plot_cell(self):

        cell_edges = []
        for e in self.half_edges:
            cell_edges.append(e.edge_points[0])
            cell_edges.append(e.edge_points[1])

        cell = shapely.LineString(cell_edges)
        
        plt.title('Our Solution')
        plt.plot(*cell.xy)
    
class Intersection:
    def __init__(self, point, half_edge):
        self.point = point
        self.half_edge = half_edge
    
    def get_half_edge(self):
        return self.half_edge
    
    def get_y(self):
        return self.point[1]

def calculate_bounding_box(points):
# Calculate smallest square that fits set of points.
    min_x = min(points, key=lambda points: points[0])[0]
    max_x = max(points, key=lambda points: points[0])[0]
    min_y = min(points, key=lambda points: points[1])[1]
    max_y = max(points, key=lambda points: points[1])[1]

    p1 = (min_x, min_y)     # Bottom left
    p2 = (min_x, max_y)     # Top left
    p3 = (max_x, max_y)     # Top right
    p4 = (max_x, min_y)     # Bottom right

    bounding_box = [p1, p2, p3, p4]
    return bounding_box

def split_points(points):
    points = sorted(points, key=lambda k: [k[0]])
    mid = (len(points) - 1) // 2
    #CITE: https://www.geeksforgeeks.org/find-median-of-list-in-python/
    #DESC: used this to figure out how to get median of list
    return points[0:mid+1], points[mid + 1:]

def calculate_linear_bisector(p, q):
    x1 = p[0]
    x2= q[0]
    y1 = p[1]
    y2 = q[1]

    #Find slope of edge pq.
    slope = (y2 - y1)/(x2 - x1)
    #Find slope of linear bisector (perpendicular to edge pq).
    m = -1/ slope

    #Calculate midpoint between a pair of points.
    midpoint = (((x1 + x2)/ 2), ((y1 + y2)/2))
    x3 = midpoint[0]
    y3 = midpoint[1]

    #Calculate linear bisector of all pairs of the current point with all other points.
    linear_bisector = (m * (x - x3)) + y3

    return linear_bisector

def merge(voronoi_l, voronoi_r):
    # Merges two voronoi diagrams together by stitching them together along
    # a constructed merge line.
    #
    # CITE: https://www.cs.princeton.edu/courses/archive/spring12/cos423/bib/vor.pdf
    # DESC: Pseudocode on pg. 6 was the inspiration for this merge implementation.

    # Construct lists of p and q potential start points.
    p_center_points = []
    q_center_points = []

    for c in voronoi_l:
        p_center_points.append(c)

    for c in voronoi_r:
        q_center_points.append(c)

    # Sort lists of p and q cell center points by y coordinates.
    p_center_points = sorted(p_center_points, key=lambda c: c[1])
    q_center_points = sorted(q_center_points, key=lambda c: c[1])

    # Start with a certain p/q pairing. If linear bisector intersects 
    # both regions, start here.
    p = (0, 0)
    q = (0, 0)
    found_valid_start_points = False

    for q in q_center_points:
        for p in p_center_points:
            # Find p and q's corresponding Voronoi cells.
            p_cell = voronoi_l[p]
            q_cell = voronoi_r[q]

            # Find linear bisector of p and q.
            linear_bisector = calculate_linear_bisector(p, q)

            # Find where linear bisector first intersects Voronoi diagrams. This will 
            # always be the same place for L and R because both are bounded by the same box.
            q_intersections = q_cell.intersection(linear_bisector) 
            p_intersections = p_cell.intersection(linear_bisector) 

            #Valid p and q's found, can exit start point finding loops.
            if p_intersections and q_intersections:
                q_entry = min(q_intersections, key=lambda q_i: q_i.get_y())
                p_entry = min(p_intersections, key=lambda p_i: p_i.get_y())
                entry = q_entry

                # Trim p and q cell edges at the entry intersection.
                q_updating_edge = q_entry.half_edge
                q_updating_edge.edge_points = (q_updating_edge.edge_points[0], q_entry.point)

                p_updating_edge = p_entry.half_edge
                p_updating_edge.edge_points = (p_entry.point, p_updating_edge.edge_points[1])

                found_valid_start_points = True

            if found_valid_start_points == True:
                break
        if found_valid_start_points == True:
            break

    if not found_valid_start_points:
        return ValueError("No valid start points")

    # Construct lists of half edges waiting to be added to p and q cells.
    p_new_edges = []
    q_new_edges = []

    # Once done creating merge line from bottom to top, becomes True.
    done_traversing = False

    # Construct merge line by looping through all relevant pairs of p and q.
    while not done_traversing:

        #Construct linear bisector of p and q.
        linear_bisector = calculate_linear_bisector(p, q)

        # Find where bisector exits p's and q's Voronoi cells.
        q_intersections = q_cell.intersection(linear_bisector)  
        q_exit = max(q_intersections, key=lambda q_i: q_i.get_y())
        p_intersections = p_cell.intersection(linear_bisector)  
        p_exit = max(p_intersections, key=lambda p_i: p_i.get_y())

        # If lowest exit point is in q (or p and q exit points are equal):
        if q_exit.get_y()  <=  p_exit.get_y():

            # Trim found q edge and q edge's twin at this intersection.
            q_updating_edge = q_exit.half_edge
            q_updating_edge.edge_points = (q_exit.point, q_updating_edge.edge_points[1])
            
            q_twin = q_exit.half_edge.twin
            if q_twin:
                q_twin.edge_points = (q_twin.edge_points[0], q_exit.point)
            
            # Make q exit point most recent exit point.
            exit = q_exit

        #If lowest exit point is in p (or p and q exit points are equal):
        if p_exit.get_y()  <=  q_exit.get_y():

            # Trim found p edge at this intersection.
            p_updating_edge = p_exit.half_edge
            p_updating_edge.edge_points = (p_updating_edge.edge_points[0], p_exit.point)
            
            p_twin = p_exit.half_edge.twin 
            if p_twin:
                p_twin.edge_points = (p_exit.point, p_twin.edge_points[1])
            
            # Make p exit point most recent exit point.
            exit = p_exit

        # Make new half edges with last entry and exit points.
        q_new_edge = HalfEdge(None, q, (entry.point, exit.point))
        p_new_edge = HalfEdge(q_new_edge, p, (exit.point, entry.point))
        q_new_edge.set_twin(p_new_edge)

        # Append new edges to p and q edge lists.
        q_new_edges.append(q_new_edge)
        p_new_edges.insert(0, p_new_edge)

        # Update q cell if has been exited.
        if q_exit.get_y()  <=  p_exit.get_y():
            q_cell.update_cell(q_new_edges)
            q_twin = q_exit.half_edge.twin
            # If q has twin, move on to bordering q cell.
            if q_twin:
                q = q_twin.center_point
                q_cell = voronoi_r[q]
                q_new_edges = []
            twin = q_twin

        # Update p cell if has been exited.
        if p_exit.get_y()  <=  q_exit.get_y():
            p_cell.update_cell(p_new_edges)
            p_twin = p_exit.half_edge.twin  
            # If p has twin, move on to bordering p cell.
            if p_twin:
                p = p_twin.center_point
                p_cell = voronoi_l[p]
                p_new_edges = []
            twin = p_twin
        
        # If no next region, merge line has been created and we are exiting 
        # the bounding box. Stop traversing.
        if not twin:
            done_traversing = True
        # Otherwise we are not done making merge line. So, make entry point 
        # for next loop around the old exit point.
        else:
            entry = exit
            entry.half_edge = exit.half_edge.twin

    # EXITS TRAVERSAL LOOP HERE
    
    # When out of while loop, we are done traversing and we have updated all 
    # relevant Voronoi cells. Output merged Voronoi diagram with left and
    # right diagram's cells.
    # CITE: https://www.geeksforgeeks.org/python-merging-two-dictionaries/
    # DESC: documentation for update() to merge two dictionaries
    merged_voronoi = {}
    merged_voronoi.update(voronoi_r)
    merged_voronoi.update(voronoi_l)
    return merged_voronoi

def divide_and_conquer(points):
    #A divide and conquer algorithm for computing Voronoi diagram. 

    if not len(points):
        return ValueError("should never have 0 points")
    
    # Base case: 1 point, return whole region
    if len(points) == 1:
        voronoi_diagram = {}        #A dictionary of point: VoronoiCell
        half_edges = []

        # Make half edges surrounding current point.
        for i in range(len(BOUNDING_BOX)):

            if i != len(BOUNDING_BOX) - 1:
                edge_points = (BOUNDING_BOX[i], BOUNDING_BOX[i+1])
            else: 
                edge_points = (BOUNDING_BOX[i], BOUNDING_BOX[0])

            half_edge = HalfEdge(None, points[0], edge_points)
            half_edges.append(half_edge)

        # Create a new Voronoi Diagram with this point's Voronoi cell as the 
        # only item in the dictionary.
        voronoi_cell = VoronoiCell(points[0], half_edges)
        voronoi_diagram[points[0]] = voronoi_cell
        return voronoi_diagram

    # Otherwise, recurse on both sides and merge them together.
    else:

        # Split set of points in half by x coordinate.
        left_points, right_points = split_points(points)

        # Recurse on left and right points to find left and right voronoi diagrams.
        left_voronoi = divide_and_conquer(left_points)
        right_voronoi = divide_and_conquer(right_points)

        # Merge left and right voronoi diagrams together.
        merged_voronoi = merge(left_voronoi, right_voronoi)
        return merged_voronoi

def main():
    points = [(1, 2), (4,7), (5,3), (2, 4)]

    global BOUNDING_BOX 
    BOUNDING_BOX = calculate_bounding_box(points)

    voronoi = divide_and_conquer(points)

    for cp in voronoi:
        plt.title('Our Solution')
        voronoi[cp].plot_cell()

    #correct voronoi diagram
    #CITE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html
    #Displays correct voronoi diagram for testing purposes
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor)
    plt.title('Correct')
    plt.show()

if __name__ == "__main__":
    main()
