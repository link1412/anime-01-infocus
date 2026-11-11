#version 410
// What is breaking under each sheet of air: the spray source averaged over the depth of a sheet, so that a line of
// surf narrower than the sheets is counted whole, wherever it lies between them.
in vec2 uv; out vec4 o;
uniform sampler2D uR1;
uniform float uSpan;     // depth of a sheet, in v of the water grid
const int N=15;
void main(){
  float s=0.;
  for(int i=0;i<N;i++) s+=textureLod(uR1,uv+vec2(0.,uSpan*((float(i)+.5)/float(N)-.5)),2.).b;
  o=vec4(s/float(N),0.,0.,1.);
}
