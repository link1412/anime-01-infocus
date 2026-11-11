#version 410
in vec2 pos; in vec2 uv0; in vec2 sd; in vec4 a; in vec3 b;
uniform vec2 uCanvas;   // layer extent in view units (period, half height)
uniform vec2 uOff;      // wrap copies
uniform vec2 uOrigin;   // canvas origin in view units
out vec2 vP, vUV, vSD; out vec4 vA; out vec3 vB;
void main(){
  vP=pos+uOff; vUV=uv0; vSD=sd; vA=a; vB=b;
  vec2 q=(vP-uOrigin)/uCanvas;
  gl_Position=vec4(q*2.-1.,0.,1.);
}
