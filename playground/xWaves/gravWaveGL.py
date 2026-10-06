import moderngl
import moderngl_window as mglw
import numpy as np

class GravitationalWaveVisualizer(mglw.WindowConfig):
    title = "Gravitational Waves Volumetric Shader"
    window_size = (900, 900)
    aspect_ratio = 1.0
    resizable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Shader Program (Vertex + Fragment Shader)
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 fragColor;
                uniform vec2 u_resolution;

                #define MAX_STEPS 120
                #define STEP_SIZE 0.03

                // Distance Estimator for Wave Shells
                float map(vec3 p) {
                    float r = length(p);
                    if (r > 4.5) return 1.0;

                    float theta = acos(p.z / (r + 1e-5));
                    float phi = atan(p.y, p.x);

                    // Quadrupole distortion & spiral wave phase
                    float wave = sin(6.0 * r - 2.0 * phi) * pow(sin(theta), 2.0);
                    float fan = sin(10.0 * phi + r * 1.5) * 0.2;
                    
                    return abs(wave + fan) - 0.15;
                }

                // Trajectory Lines Distance Function
                float distanceToLine(vec3 p, vec3 a, vec3 b) {
                    vec3 pa = p - a, ba = b - a;
                    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
                    return length(pa - ba * h);
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;

                    // Camera position & setup
                    vec3 ro = vec3(5.0 * sin(0.8), 3.0, 5.0 * cos(0.8));
                    vec3 ta = vec3(0.0, 0.0, 0.0);
                    vec3 ww = normalize(ta - ro);
                    vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
                    vec3 vv = cross(uu, ww);
                    vec3 rd = normalize(uv.x * uu + uv.y * vv + 1.3 * ww);

                    // Raymarching Volumetric Accumulation
                    vec3 accumColor = vec3(0.0);
                    float alphaAcc = 0.0;
                    float t = 0.5;

                    for (int i = 0; i < MAX_STEPS; i++) {
                        vec3 pos = ro + rd * t;
                        float r = length(pos);

                        if (r < 4.5) {
                            float d = map(pos);
                            
                            if (d < 0.1) {
                                float density = smoothstep(0.1, 0.0, d) * exp(-0.35 * r);
                                
                                // Color Map Logic: Core Orange -> Cyan -> Outer Deep Blue
                                vec3 color;
                                if (r < 0.9) {
                                    color = mix(vec3(1.0, 0.2, 0.0), vec3(1.0, 0.6, 0.1), r / 0.9); // Orange Core
                                } else if (r < 2.5) {
                                    float factor = (r - 0.9) / 1.6;
                                    color = mix(vec3(0.0, 0.9, 1.0), vec3(0.0, 0.4, 0.8), factor); // Bright Cyan
                                } else {
                                    float factor = clamp((r - 2.5) / 2.0, 0.0, 1.0);
                                    color = mix(vec3(0.0, 0.4, 0.8), vec3(0.01, 0.05, 0.3), factor); // Deep Blue
                                }

                                // Fresnel Glow Effect at shell edges
                                float edgeGlow = pow(1.0 - abs(dot(rd, normalize(pos))), 2.0);
                                color += vec3(0.2, 0.6, 1.0) * edgeGlow * 0.5;

                                float stepAlpha = density * 0.08;
                                accumColor += color * stepAlpha * (1.0 - alphaAcc);
                                alphaAcc += stepAlpha;

                                if (alphaAcc >= 0.95) break;
                            }
                        }
                        t += STEP_SIZE;
                    }

                    // Render Trajectory Lines (Hyperbolic Ray Lines)
                    vec3 line1A = vec3(-3.0, -1.5, -4.0), line1B = vec3(1.5, 3.0, 4.0);
                    vec3 line2A = vec3(-1.0, -3.0, -4.0), line2B = vec3(2.5, 1.5, 4.0);
                    
                    float dLine1 = distanceToLine(ro + rd * 4.0, line1A, line1B);
                    float dLine2 = distanceToLine(ro + rd * 4.0, line2A, line2B);

                    if (dLine1 < 0.015) accumColor += vec3(0.9) * (1.0 - dLine1 / 0.015);
                    if (dLine2 < 0.015) accumColor += vec3(0.9) * (1.0 - dLine2 / 0.015);

                    fragColor = vec4(accumColor, 1.0);
                }
            '''
        )

        # Full-screen Quad Setup
        vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

        # Uniforms
        self.prog['u_resolution'].value = self.window_size

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

if __name__ == '__main__':
    mglw.run_window_config(GravitationalWaveVisualizer)