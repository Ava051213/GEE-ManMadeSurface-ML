import rasterio
import rasterio.warp

#old_left, old_bottom, old_right, old_top = -2159931, 1651586, -2020559, 1819574
old_left, old_bottom, old_right, old_top = 265454, 4327166, 265754, 4327466 #394410,6158823,395050,6159463 
#457274.0,4328603.0,457530.0,4328859.0 #443833.85873601056,5368019.6957511455,444474.14571742853,5368659.69766417 #-118.23960212741923,34.00720407534399,-118.23897985492778,34.00804009348488
#443833.85873601056,5368019.6957511455,444474.14571742853,5368659.69766417
old_crs = 'EPSG:32615' # 'EPSG:4326'
#new_crs = 'EPSG:4326'
new_crs = 'EPSG:4326' #'EPSG:3857'
#new_crs = 'EPSG:3857' #'EPSG:3857'

left, bottom, right, top = rasterio.warp.transform_bounds(old_crs, new_crs, old_left, old_bottom, old_right, old_top)

print(str(left)+','+ str(bottom)+','+str(right)+','+str(top))
#print(str(top)+','+ str(left)+','+str(bottom)+','+str(right))


tile_left, tile_bottom, tile_right, tile_top = -75.49423379074392, 39.105148315429688, -75.491137661740822, 39.107894897460938
new_left, new_bottom, new_right, new_top = rasterio.warp.transform_bounds('EPSG:4326', 'EPSG:26918', tile_left, tile_bottom, tile_right, tile_top)

print((old_left-new_left)*1.0/(new_right-new_left))
print((new_right-old_right)*1.0/(new_right-new_left))
print((new_top-old_top)/(new_top-new_bottom))
print((old_bottom-new_bottom)/(new_top-new_bottom))
