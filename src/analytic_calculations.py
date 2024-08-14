import sympy as sp
from IPython.display import display, Math

def spherical_to_cartesian(rho, phi, theta):
    x = rho * sp.sin(phi) * sp.cos(theta)
    y = rho * sp.sin(phi) * sp.sin(theta)
    z = rho * sp.cos(phi)
    return [x, y, z]

# define the chip tilt in terms of a normal vector to the surface
phi, theta = sp.symbols('φ θ')

# define the width and height of the chip
w, h = sp.symbols('w h')

# define the four vectors to the chip corners
A, B, C, D = sp.symbols(r'\mathbf{\A}_1 \mathbf{\B} \mathbf{\C} \mathbf{\D}')

# define the length of the vectors in terms of rho
rho = sp.symbols('ρ')
rho_ex = sp.sqrt((h/2)**2 + (w/2)**2)

# make the normal vector
n_c = sp.Matrix(spherical_to_cartesian(1, phi, theta))

# define the A vector
phi_a = phi + sp.pi/2
theta_a = theta
A = sp.Matrix(spherical_to_cartesian(rho, phi_a, theta_a))

print(sp.latex(A))


