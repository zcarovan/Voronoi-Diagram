"""
    File:        brute_voronoi.py
    Author:      Zoe Carovano
    Course:      CS 330 - Algorithms
    Semester:    Spring 2024
    Assignment:  Voronoi Diagram Brute Force Implementation
"""

import math
import sympy as sympy
from sympy.abc import x, y
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import shapely.geometry

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

def on_left(a, b, c):
# Checks if point c is on left of line segment ab.
    return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]) > 0

def update_region(point, prev_region, bisector, bounding_box):
# Updates point's Voronoi cell given a new linear bisector.

    line = []
    intersection_points = []

    #See where bisector intersects bounding box.
    for i in range(len(bounding_box)):

        #If odd, evaluate what y is by plugging in x_coordinate of line.
        if i % 2:
            #See where bisector intersects bounding line's x coordinate.
            intersection_point = bounding_box[i][0], bisector.evalf(3, subs={x: bounding_box[i][0]})
            # If in y range of bounding edge, add to intersection points.
            if bounding_box[0][1] <= intersection_point[1] <= bounding_box[1][1]:
                intersection_points.append(intersection_point)
                line.append(intersection_point)

        #If even, evaluate what x is by plugging in y_coordinate of line.
        else: 
           #See where bisector intersects bounding line's y coordinate.
            intersection_point = sympy.solve(sympy.Eq(bisector, bounding_box[i][1]), x)[0].evalf(3), bounding_box[i][1]
            # If in x range of bounding edge, add to intersection points.
            if bounding_box[0][0] < intersection_point[0] < bounding_box[2][0]:
                intersection_points.append(intersection_point)
                line.append(intersection_point)

    bisector_area = bounding_box[:]

    #If point on left of bisector, include all points from region that are to the left of line.
    if line: 
        if on_left(line[0], line[-1], point):
            for b in bounding_box:
                if not on_left(line[0], line[-1], b):
                    bisector_area.remove(b)

    #If point on right of bisector, include all points from region that are to the right of line.
        else:
            for b in bounding_box:
                if on_left(line[0], line[-1], b):
                    bisector_area.remove(b)

    if len(intersection_points) == 2:
        bisector_area.append(intersection_points[0])
        bisector_area.append(intersection_points[1])

    #Sort bisector region points in counterclockwise order.
    cent = (sum([b[0] for b in bisector_area])/ len(bisector_area),sum([b[1] for b in bisector_area])/len(bisector_area))
    bisector_area.sort(key=lambda b: math.atan2(b[1]-cent[1],b[0]-cent[0]), reverse = True)

    #Does the bisector cut into the current region?
    prev_region = shapely.Polygon(prev_region)
    bisector_region = shapely.Polygon(bisector_area)
    
    # If intersection, return intersection of bisector region and previous region 
    # as the new region
    if prev_region.intersects(bisector_region):
        new_region = prev_region.intersection(bisector_region)
        return new_region
    
    #If there are no intersection points, return previous region as region.
    else:
        region = prev_region 
        return prev_region
        

def brute_voronoi(points):
# A brute force method to compute Voronoi diagram of set of points.

    # Calculate the finite region you are looking at.
    bounding_box = calculate_bounding_box(points)
    voronoi_cells = []

    #For all pairs of points:
    for p in points:
        # Each point has an associated region (Voronoi cell)
        region = bounding_box
        for q in points:
            if p != q:
                linear_bisector = calculate_linear_bisector(p, q)
                region = update_region(p, region, linear_bisector, bounding_box)
        voronoi_cells.append((p, region))
        plt.title('Our Solution')
        plt.plot(*region.exterior.xy)

def main():
    points = [(1, 2), (4,7), (5,3), (7,5), (2, 4)]
    brute_voronoi(points)

    #correct voronoi diagram
    #CITE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html
    #Displays correct voronoi diagram for testing purposes
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor)
    plt.title('Correct')
    plt.show()

if __name__ == "__main__":
    main()
