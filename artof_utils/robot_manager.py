from os import path
import time

from artof_utils.singleton import Singleton
from artof_utils.schemas.settings import load_settings
from artof_utils.field_manager import FieldManager  
from artof_utils.schemas.hitches import Hitches
from artof_utils.schemas.navigation import Navigation
from artof_utils.schemas.state import State
from artof_utils.gis import hardware as hw
from artof_utils.gis import shape as shp
from artof_utils.gis import polygon
from artof_utils.schemas.settings import AutoMode
from artof_utils.redis_manager import redis_manager
from artof_utils.visualisation_manager import visualisation_manager
from shapely.geometry import Point
import artof_utils.paths as paths

def get_current_field_name():
    name = redis_manager.get_value('pc.field.name')
    print("Current field name: %s" % name)
    return name if name else ''


class RobotManager(metaclass=Singleton):
    """
    A singleton class responsible for managing the robot's settings, field information,
    navigation, and interaction with external resources like redis_manager.
    """

    def __init__(self):
        self.platform_settings = None
        self.field = None

        if paths.loaded:
            self.load_settings()

        self.hitches = Hitches()
        self.navigation = Navigation()

        if self.platform_settings:
            self.hitches.add_hitches(self.platform_settings.hitches)

        all_keys = redis_manager.variables.keys()
        self.robot_state_vars = [key for key in all_keys if key.startswith('plc.monitor.state.')]

    def load_settings(self):
        print("Load settings")
        self.platform_settings = load_settings()

    def load_field(self, gdf=None):
        print("Load Field ")
        self.field = FieldManager(get_current_field_name(), gdf)
        as_applied = path.join(paths.fields, self.field.info.name, 'rasters')
        visualisation_manager.initialize_field(self.field.info.bounds, resolution=0.000001, as_applied_path=as_applied)
        visualisation_manager.load_static_layers(field=self.field)

       
    @staticmethod
    def get_navigation_modes():
        if robot_manager.platform_settings is None:
            return [(1, 'pp 90\u00b0 turn'), (2, 'pp 180\u00b0 turn'), (3, 'pure pp'), (4, 'pp rollback'), (5, 'external')]
        else:
            return [(mode.id, mode.name) for mode in robot_manager.platform_settings.nav_modes]

    def get_navigation_states(self):
        if self.platform_settings is None:
            auto_modes_settings = [AutoMode.model_validate({'name': 'normal', 'id': 0}), AutoMode.model_validate({'name': 'auto', 'id': 1})]
        else:
            auto_modes_settings = [AutoMode.model_validate({'name': 'normal', 'id': 0})] + self.platform_settings.auto_modes
        state_names = [auto_mode_setting.name for auto_mode_setting in auto_modes_settings]
        return state_names

    def get_navigation_state(self):
        state_vars = redis_manager.get_n_values(self.robot_state_vars)
        active_states = [k.replace('plc.monitor.state.', '') for k, v in state_vars.items() if v]
        current_state = active_states[0] if len(active_states) > 0 else robot_manager.get_navigation_states()[0]

        if self.get_simulation_mode() and self.get_simulation_auto():
            current_state = 'auto'

        return current_state

    def set_navigation_state(self, navigation_state):
        sim_mode = self.get_simulation_mode()
        if sim_mode:
            redis_manager.set_value('pc.simulation.auto', navigation_state != 'normal')
        else:
            if navigation_state in self.get_navigation_states():
                redis_manager.set_value('plc.monitor.state.' + navigation_state, True)

        control_name = 'plc.control.state.' + navigation_state
        if navigation_state in self.get_navigation_states() and control_name in redis_manager.variables.keys():
            redis_manager.set_value(control_name, True)
            time.sleep(0.5)
            redis_manager.set_value(control_name, False)
            print("Pulsed %s" % control_name)

    def set_position_latlon(self, lat, lon):
        wgs84_crs = 'EPSG:4326'  
        if self.platform_settings and self.platform_settings.gps:
            utm_crs = 'EPSG:326%d' % self.platform_settings.gps.utm_zone
        else:
            utm_crs = 'EPSG:32631'  
        x, y = shp.transform_crs(wgs84_crs, utm_crs, [lat, lon])
        
        self.set_position(x, y)

    @staticmethod
    def set_position(x, y, yaw=None):
        robot_ref_state = redis_manager.get_json_value("robot.ref.state")
        if robot_ref_state is None:
            return
        robot_ref_state["T"] = [float(x), float(y), 0.0]
        if yaw is not None:
            robot_ref_state["R"] = [0.0, 0.0, yaw]
        redis_manager.set_json_value("robot.ref.state", robot_ref_state)
        
    @staticmethod
    def set_velocity(vx, omega):
        redis_manager.set_value('plc.control.navigation.velocity.longitudinal', vx)
        redis_manager.set_value('plc.control.navigation.velocity.angular', omega)

    @staticmethod
    def get_velocity():
        vx = redis_manager.get_value('plc.control.navigation.velocity.longitudinal')
        omega = redis_manager.get_value('plc.control.navigation.velocity.angular')
        return vx, omega

    @staticmethod
    def get_simulation_mode():
        return redis_manager.get_value('pc.simulation.active')

    @staticmethod
    def get_simulation_auto():
        return redis_manager.get_value('pc.simulation.auto')

    def set_simulation_mode(self, active=True):
        redis_manager.set_value('pc.simulation.active', active)
        
        if active and self.field.traject_manager.exists:
            traject_gdf = self.field.gdf[self.field.gdf['name'] == 'traject']
            
            if not traject_gdf.empty:
                traject_geom = traject_gdf.geometry.iloc[0]
                if hasattr(traject_geom, 'coords') and len(traject_geom.coords) >= 2:
                    traject_points = list(traject_geom.coords)
                    
                    first_point = Point(traject_points[0])
                    second_point = Point(traject_points[1])
                    path_orientation = shp.get_orientation(first_point, second_point)

                    lon = first_point.x
                    lat = first_point.y

                    self.set_position_latlon(lat, lon)
                    
                    robot_ref_state = redis_manager.get_json_value("robot.ref.state")
                    if robot_ref_state is not None:
                        robot_ref_state["R"] = [0.0, 0.0, path_orientation]
                        redis_manager.set_json_value("robot.ref.state", robot_ref_state)
                    
        if not active:
            self.set_navigation_state('normal')

    @staticmethod
    def set_simulation_speed_factor(factor):
        redis_manager.set_value('pc.simulation.factor', factor)

    @staticmethod
    def get_simulation_speed_factor():
        return redis_manager.get_value('pc.simulation.factor')

    @staticmethod
    def get_programming_mode():
        return redis_manager.get_value('plc.monitor.substate.programming')

    @staticmethod
    def acknowledge_notification():
        redis_manager.set_value('pc.execution.notification', '-')

    @staticmethod
    def update_field():
        redis_manager.set_value('pc.field.updated', True)

    def status(self):
        return redis_manager.get_json_value("robot.status")

    def context(self):
        redis_data = redis_manager.get_n_json_value(["robot.center.state", "robot.ref.state", "robot.head.state",
                                                    "robot.contour", "hitch.states", "implement.states",
                                                    "navigation.controller.info"])
        r = dict()
        r['robot'] = {'contours': redis_data['robot.contour'],
                      'center': redis_data['robot.center.state']['point'],
                      'ref': redis_data['robot.ref.state']['point'],
                      'head': redis_data['robot.head.state']['point'],
                      'orientation': redis_data['navigation.controller.info']['heading']}
        r['hitches'] = redis_data['hitch.states']
        r['implements'] = redis_data['implement.states']
        r['controller_info'] = redis_data['navigation.controller.info']

        try:
            robot_contour = redis_data.get('robot.contour')
            implement_data = r.get('implements', {})
            
            visualisation_manager.process_new_state(
                implement_data=implement_data, 
                robot_contour=robot_contour,
            )
        except Exception as e:
            print(f"[RobotManager] Vis-update error: {e}")

        return r

        return r

robot_manager = RobotManager()