using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Points;

/// <summary>
///    _   _ _   _ _     
///   | | | | |_(_) |___ 
///   | | | | __| | / __|
///   | |_| | |_| | \__ \
///    \___/ \__|_|_|___/
///                      
/// Maths related utility functions.
/// </summary>
internal static class MathUtils
{
    /// <summary>
    /// Logic requires radians but we track angles in degrees, this converts.
    /// </summary>
    /// <param name="angle"></param>
    /// <returns></returns>
    internal static double DegreesInRadians(double angle)
    {
        return Math.PI * angle / 180;
    }

    /// <summary>
    /// Ensures value is between the min and max.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="val"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <returns></returns>
    internal static T Clamp<T>(this T val, T min, T max) where T : IComparable<T>
    {
        if (val.CompareTo(min) < 0)
        {
            return min;
        }

        if (val.CompareTo(max) > 0)
        {
            return max;
        }

        return val;
    }
}