#version 410
// a raindrop crossing the frame during the shutter: a line of what it sees, one frame long
in vec2 vS; in vec3 vB; in float vL;
out vec4 o;
uniform vec3 uTint;
void main(){
  float dx=max(abs(vS.x)-vL,0.);
  float d=length(vec2(dx,vS.y))/vB.x;
  float a=smoothstep(1.,1.-vB.z,d);
  // brighter down one edge: it is a lens, the sun is on that side
  float e=.72+.28*vS.y/vB.x;
  o=vec4(uTint*a*vB.y*e,0.);
}
