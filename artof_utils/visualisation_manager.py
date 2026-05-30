from os import path
import time
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from shapely.geometry.base import BaseGeometry
from shapely.geometry import LineString, MultiPoint, Polygon
import artof_utils.paths as paths
import threading
import multiprocessing as mp
import queue
from PIL import Image
import io
import os
import geopandas as gpd
from artof_utils.redis_manager import redis_manager

from collections import defaultdict

class CoreVisualisationManager:
    def __init__(self):
        self.bounds = None
        self.resolution = None
        self.lock = threading.RLock()
        self.latest_frame = None
        self.prev_sections = {}
        self.prev_full_width = None
        self.last_save_time = time.time()
        self.as_applied_filepath = None
        self.was_active = False
        self.saved_while_stopped = False
        self.last_robot_contour_coords = None
        self.last_covered_tick = None
        self.tick = 0


        # color lut
        self.lut = np.zeros((256, 4), dtype=np.uint8)
        self.lut[1] = [150, 150, 150, 100]  
        
        for i in range(2, 256):
            t = (i - 2) / 253.0 
            if t < 0.5:
                self.lut[i] = [int(t * 2 * 255), 255, 0, 180]
            else:
                self.lut[i] = [255, int((1.0 - t) * 2 * 255), 0, 180]
        
    def initialize_field(self, bounds, resolution, as_applied_path):
        self.bounds = bounds
        self.resolution = resolution
        minx, miny, maxx, maxy = self.bounds
        self.width = max(1, int(np.ceil((maxx - minx) / self.resolution)))
        self.height = max(1, int(np.ceil((maxy - miny) / self.resolution)))
        self.transform = from_origin(minx, maxy, self.resolution, self.resolution)

        self.static_map = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.live_map = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.as_applied_data = np.zeros((self.height, self.width), dtype=np.uint8)
        self.last_covered_tick = np.full((self.height, self.width), -100, dtype=np.int32)
        self.tick = 0

        self.prev_sections.clear()
        self.prev_full_width = None
        self.as_applied_filepath = as_applied_path + "/as_applied.tiff"
        os.makedirs(os.path.dirname(as_applied_path), exist_ok=True)
        self.last_robot_contour_coords = None

    def load_static_layers(self, field):
        gdf = field.gdf
        folder_path = os.path.join(paths.fields, field.name)
        
        with self.lock:
            self.static_map.fill(0)

            from artof_utils.gis.raster import Raster as rstr

            # Determine field bounds from geofence for raster generation
            field_bounds = None
            if gdf is not None and not gdf.empty:
                gf_row = gdf[gdf['name'] == 'geofence']
                if not gf_row.empty and gf_row.geometry.iloc[0] is not None:
                    field_bounds = gf_row.geometry.iloc[0].bounds

            def generate_and_save_raster(name, geometry):
                """Generate a raster from geometry using existing Raster utility and save to disk."""
                if geometry is None or geometry.is_empty or field_bounds is None:
                    return None
                try:
                    raster_rel = f"rasters/{name}.tif"
                    full_path = os.path.join(folder_path, raster_rel)
                    if not os.path.exists(full_path):
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        raster_array, transform, width, height = rstr.generate_array(
                            geometry=geometry,
                            bounds=field_bounds,
                            resolution=self.resolution,
                        )
                        with rasterio.open(
                            full_path, 'w', driver='GTiff',
                            height=height, width=width,
                            count=1, dtype=raster_array.dtype,
                            crs='EPSG:4326', transform=transform, compress='lzw'
                        ) as dst:
                            dst.write(raster_array, 1)
                    return raster_rel
                except Exception:
                    return None

            def overlay_tiff(rel_path, color, is_geofence=False, geometry=None, name=None):
                # If no raster file: try to generate it from the geometry
                if (not isinstance(rel_path, str) or not rel_path) and geometry is not None and name:
                    rel_path = generate_and_save_raster(name, geometry)
                if not isinstance(rel_path, str) or not rel_path:
                    return
                full_path = os.path.join(folder_path, rel_path)
                if not os.path.exists(full_path):
                    return
                try:
                    with rasterio.open(full_path) as src:
                        data = src.read(1)
                        mask = data == 255
                        if is_geofence:
                                eroded = mask.copy()
                                eroded[1:, :] &= mask[:-1, :]
                                eroded[:-1, :] &= mask[1:, :]
                                eroded[:, 1:] &= mask[:, :-1]
                                eroded[:, :-1] &= mask[:, 1:]
                                mask = mask & ~eroded
                                thick_border = mask.copy()
                                thick_border[1:, :] |= mask[:-1, :]
                                thick_border[:-1, :] |= mask[1:, :]
                                thick_border[:, 1:] |= mask[:, :-1]
                                thick_border[:, :-1] |= mask[:, 1:]
                                mask = thick_border
                        self.static_map[mask] = color
                except Exception as e:
                    pass

            if gdf is not None and not gdf.empty:
                raster_col = 'raster_source' if 'raster_source' in gdf.columns else None
                task_colors = [
                    [0, 150, 255, 60],    # blauw
                    [255, 100, 180, 60],  # roze
                    [160, 80, 220, 60],   # paars
                    [0, 200, 180, 60],    # turkoois
                    [100, 200, 255, 60],  # lichtblauw
                ]
                for task_idx, (_, row) in enumerate(gdf[gdf['type'] == 'task'].iterrows()):
                    color = task_colors[task_idx % len(task_colors)]
                    overlay_tiff(row.get(raster_col) if raster_col else None,
                                 color,
                                 geometry=row.geometry, name=row.get('name'))
                for _, row in gdf[gdf['name'] == 'traject'].iterrows():
                    overlay_tiff(row.get(raster_col) if raster_col else None,
                                 [255, 0, 20, 255],
                                 geometry=row.geometry, name='traject')
                for _, row in gdf[gdf['name'] == 'geofence'].iterrows():
                    overlay_tiff(row.get(raster_col) if raster_col else None,
                                 [255, 0, 0, 255], is_geofence=True,
                                 geometry=row.geometry, name='geofence')

            if hasattr(self, 'as_applied_filepath') and self.as_applied_filepath and os.path.exists(self.as_applied_filepath):
                try:
                    with rasterio.open(self.as_applied_filepath) as src:
                        loaded_data = src.read(1)
                        if loaded_data.shape == self.as_applied_data.shape:
                            self.as_applied_data = loaded_data
                except Exception:
                    pass
            
            self.live_map = self.static_map.copy()
            
            mask = self.as_applied_data > 0
            self.live_map[mask] = self.lut[self.as_applied_data[mask]]
            
            self._generate_binary_frame_no_lock()

    def process_new_state(self, implement_data, robot_contour=None):
        """Returnt True als er een nieuwe map is gerenderd, False als er niets is gebeurd."""
        if not hasattr(self, 'static_map'): return False
        
        val_l = redis_manager.get_value("plc.monitor.navigation.velocity.longitudinal")
        val_a = redis_manager.get_value("plc.monitor.navigation.velocity.angular")
        current_velocity_l = float(val_l) if val_l is not None else 0.0
        current_velocity_a = float(val_a) if val_a is not None else 0.0
        
        is_moving = (abs(current_velocity_l) + abs(current_velocity_a)) > 0.01

        # Save when robot stops
        if not is_moving:
            if not getattr(self, 'saved_while_stopped', False):
                if hasattr(self, 'as_applied_filepath') and self.as_applied_filepath:
                    self.last_save_time = time.time()
                    threading.Thread(target=self.save_as_applied_to_disk, args=(self.as_applied_filepath,), daemon=True).start()
                self.saved_while_stopped = True
            return False

        is_active = False
        if implement_data and isinstance(implement_data, dict):
            for imp_name, implement in implement_data.items():
                if 'sections' in implement:
                    for i, section in enumerate(implement['sections']):
                        rb = redis_manager.get_value(f"plc.monitor.hitch_rb.feedback_sections.{i}")
                        fb = redis_manager.get_value(f"plc.monitor.hitch_fb.feedback_sections.{i}")
                        rb_int = int(float(rb)) if rb is not None else 0
                        fb_int = int(float(fb)) if fb is not None else 0
                        if max(rb_int, fb_int) > 0:
                            is_active = True
                            break

        try:
            r_coords = None
            if robot_contour:
                contour_list = robot_contour.get('latlng') if isinstance(robot_contour, dict) else robot_contour
                if isinstance(contour_list, list) and len(contour_list) >= 3:
                    r_coords = [(pt[1], pt[0]) if isinstance(pt, (list, tuple)) else (pt.get('lng', pt.get('lon')), pt.get('lat')) for pt in contour_list]

            self.last_robot_contour_coords = r_coords

            path_polygons = []    
            dose_polygons = defaultdict(list)
            
            current_sections = {}
            current_implement_geoms = []

            if implement_data and isinstance(implement_data, dict):
                for imp_name, implement in implement_data.items():

                        all_sec_coords = []
                        for i, section in enumerate(implement['sections']):
                            sec_id = f"{imp_name}_{i}"
                            coords = [(pt[1], pt[0]) for pt in section.get('latlng', [])]
                            if len(coords) < 3: continue
                            
                            all_sec_coords.extend(coords)
                            current_implement_geoms.append(Polygon(coords))
                            
                            # Use actual Redis feedback as draw condition.
                            val_fb = redis_manager.get_value(f"plc.monitor.hitch_fb.feedback_sections.{i}")
                            val_rb = redis_manager.get_value(f"plc.monitor.hitch_rb.feedback_sections.{i}")
                            fb_int = int(float(val_fb)) if val_fb is not None else 0
                            rb_int = int(float(val_rb)) if val_rb is not None else 0
                            feedback_val = max(0, min(255, max(fb_int, rb_int)))

                            if feedback_val > 0:
                                curr_geom = Polygon(coords)
                                mapped_val = int((feedback_val / 255.0) * 253) + 2
                                dose_polygons[mapped_val].append(curr_geom)

                                if is_moving and sec_id in self.prev_sections:
                                    prev_coords = self.prev_sections[sec_id]
                                    sweep_geom = MultiPoint(prev_coords + coords).minimum_rotated_rectangle
                                    dose_polygons[mapped_val].append(sweep_geom)

                            current_sections[sec_id] = coords

            if all_sec_coords:
                if is_moving or is_active:
                    path_polygons.append(Polygon(all_sec_coords))
                    if is_moving and self.prev_full_width:
                        path_sweep = [
                            self.prev_full_width[0], self.prev_full_width[-1],
                            all_sec_coords[-1], all_sec_coords[0]
                        ]
                        path_polygons.append(Polygon(path_sweep))
                self.prev_full_width = all_sec_coords

            if is_moving or is_active:
                self.prev_sections.update(current_sections)

            robot_geom = None
            if r_coords and len(r_coords) >= 3:
                robot_geom = Polygon(r_coords)

            map_changed = bool(path_polygons or dose_polygons)

            with self.lock:
                if path_polygons:
                    path_raster = rasterize([(g, 1) for g in path_polygons], out_shape=(self.height, self.width), 
                                            transform=self.transform, fill=0, all_touched=True, dtype='uint8')
                    self.as_applied_data = np.maximum(self.as_applied_data, path_raster)

                for val, geoms in dose_polygons.items():
                    val_raster = rasterize([(g, val) for g in geoms], out_shape=(self.height, self.width), 
                                           transform=self.transform, fill=0, all_touched=True, dtype='uint8')
                    covered = val_raster > 0
                    if self.last_covered_tick is None:
                        self.last_covered_tick = np.full((self.height, self.width), -100, dtype=np.int32)
                    # Only add to pixels not covered in the previous 3 ticks 
                    new_pass = covered & (self.last_covered_tick < self.tick - 3)
                    delta = val_raster * new_pass.astype(np.uint8)
                    accumulated = self.as_applied_data.astype(np.uint16) + delta.astype(np.uint16)
                    self.as_applied_data = np.clip(accumulated, 0, 255).astype(np.uint8)
                    self.last_covered_tick[covered] = self.tick
                self.tick += 1

                # Only regenerate PNG when as-applied data actually changed.
                # Robot and implement contours are drawn as vectors in the frontend.
                if map_changed:
                    self.live_map = self.static_map.copy()
                    mask = self.as_applied_data > 0
                    self.live_map[mask] = self.lut[self.as_applied_data[mask]]
                    self._generate_binary_frame_no_lock()

            self.saved_while_stopped = False
            if self.as_applied_filepath and (time.time() - self.last_save_time > 5.0):
                self.last_save_time = time.time()
                threading.Thread(target=self.save_as_applied_to_disk, args=(self.as_applied_filepath,), daemon=True).start()

            return map_changed

        except Exception as e:
            print(f"Error in process_new_state: {e}")
            return False

    def _generate_binary_frame_no_lock(self):
        img = Image.fromarray(self.live_map, mode='RGBA')
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", compress_level=1) 
        self.latest_frame = buffer.getvalue() 

    def get_latest_frame(self):
        return self.latest_frame

    def save_as_applied_to_disk(self, filepath):
        with self.lock:
            with rasterio.open(
                filepath, 'w', driver='GTiff', height=self.height, width=self.width,
                count=1, dtype=self.as_applied_data.dtype, crs='EPSG:4326',
                transform=self.transform, compress='lzw'
            ) as dst:
                dst.write(self.as_applied_data, 1)

    def reset_and_archive_as_applied(self):
        with self.lock:
            print("Resetting task and archiving as-applied data...")
            if not self.as_applied_filepath or not os.path.exists(self.as_applied_filepath):
                self.as_applied_data.fill(0)
                self._generate_binary_frame_no_lock()
                return

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            directory = os.path.dirname(self.as_applied_filepath)
            filename = os.path.basename(self.as_applied_filepath)
            name, ext = os.path.splitext(filename)
            archived_path = os.path.join(directory, f"{name}_{timestamp}{ext}")

            try:
                self.save_as_applied_to_disk(self.as_applied_filepath)
                os.rename(self.as_applied_filepath, archived_path)
                
                self.as_applied_data.fill(0)
                self._generate_binary_frame_no_lock()
                self.save_as_applied_to_disk(self.as_applied_filepath)
            except Exception as e:
                print(f"Error archiveren: {e}")

                
class VisualisationManager:
    def __init__(self):
        self.input_q = mp.Queue()
        self.output_q = mp.Queue()
        self.latest_frame = None

        # process for rasterize
        self.process = mp.Process(target=self._worker_loop, args=(self.input_q, self.output_q), daemon=True)
        self.process.start()

        # thread for file i/o
        self.reader_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.reader_thread.start()

    def initialize_field(self, bounds, resolution, as_applied_path):
        self.input_q.put(('INIT', bounds, resolution, as_applied_path))

    def load_static_layers(self, field):
        gdf_json = field.gdf.to_json() if (field.gdf is not None and not field.gdf.empty) else None
        self.input_q.put(('LOAD_STATIC', field.name, gdf_json))

    def process_new_state(self, implement_data, robot_contour=None):
        self.input_q.put(('UPDATE', implement_data, robot_contour))

    def get_latest_frame(self):
        return self.latest_frame

    def _read_frames(self):
        while True:
            try:
                frame = self.output_q.get()
                if frame:
                    self.latest_frame = frame
            except Exception:
                pass

    def reset_task(self):
        print("Reset task command received in VisualisationManager.")
        self.input_q.put(('RESET_TASK',))

    @staticmethod
    def _worker_loop(in_q, out_q):
        core = CoreVisualisationManager()
        
        # dummy class for load_static_layers
        class DummyField:
            def __init__(self):
                self.name = ""
                self.gdf = None

        while True:
            try:
                msg = in_q.get()
                cmd = msg[0]
                
                if cmd == 'INIT':
                    core.initialize_field(msg[1], msg[2], msg[3])
                    
                elif cmd == 'LOAD_STATIC':
                    field_name = msg[1]
                    gdf_json = msg[2]
                    dummy = DummyField()
                    dummy.name = field_name
                    if gdf_json:
                        dummy.gdf = gpd.read_file(io.StringIO(gdf_json))
                    core.load_static_layers(dummy)
                    out_q.put(core.get_latest_frame())
                    
                elif cmd == 'UPDATE':
                    latest_msg = msg
                    
                    while not in_q.empty():
                        try:
                            next_msg = in_q.get_nowait()
                            if next_msg[0] == 'UPDATE':
                                latest_msg = next_msg
                        except Exception:
                            break

                    # Only render when there are changes
                    has_new_frame = core.process_new_state(latest_msg[1], latest_msg[2])
                    
                    if has_new_frame:
                        while not out_q.empty():
                            try: out_q.get_nowait()
                            except: pass
                            
                        frame = core.get_latest_frame()
                        if frame:
                            out_q.put(frame)

                elif cmd == 'RESET_TASK':
                    print("Processing reset task command in worker loop...")
                    core.reset_and_archive_as_applied()
                    out_q.put(core.get_latest_frame())
                        
            except Exception as e:
                print(f"[Visualisation Worker Error] {e}")

visualisation_manager = VisualisationManager()