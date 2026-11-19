from tests.support import SourceCase
from scipy.io import netcdf_file
import io,json
class Contract(SourceCase):
    def array(self,q):
        s,h,b=self.request("/array.nc?"+q);self.assertEqual(s,200)
        return netcdf_file(io.BytesIO(b),mmap=False)
    def test_catalog_dimensions(self):
        c=json.loads(self.request("/catalog")[2])
        self.assertEqual(c["variables"]["current"]["dimensions"],["station","sample"])
        self.assertEqual(c["axes"]["station"],[30,20,10])
    def test_actual_netcdf_coordinates(self):
        with self.array("variable=temperature&time=0:2&station=1:3") as f:
            self.assertEqual(f.variables["temperature"].shape,(2,2))
            self.assertEqual(f.variables["station"][:].tolist(),[20,10])
    def test_limit(self):
        self.assertEqual(self.request("/array.nc?variable=temperature")[0],413)
    def test_empty_selection(self):
        self.assertEqual(self.request("/array.nc?variable=temperature&time=0:0&station=0:1")[0],204)
    def test_quality_not_coverage(self):
        with self.array("variable=temperature&time=1:2&station=1:2") as f:
            self.assertEqual(f.variables["temperature"][:].item(),-9999)
            self.assertEqual(f.variables["quality"][:].item(),0)
    def test_axis_insertion_not_append(self):
        self.control(1)
        c=json.loads(self.request("/catalog")[2])
        self.assertEqual(c["axes"]["time"][3],2.5)
        with self.array("variable=temperature&version=r1&time=3:4&station=0:1") as f:self.assertEqual(f.variables["time"][:].item(),3)
    def test_retirement(self):
        self.control(2)
        self.assertEqual(self.request("/array.nc?variable=current&version=r1&station=0:1&sample=0:1")[0],410)
        self.assertEqual(self.request("/catalog?version=r2")[0],200)
    def test_out_of_bounds(self):
        self.assertEqual(self.request("/array.nc?variable=current&station=0:1&sample=0:99")[0],400)
