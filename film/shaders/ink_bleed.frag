#version 410
// Bake-time wet-in-wet: ink moves only where paper is wet, along conductive fibres; the water
// front itself creeps. r=ink, g=reserve (untouched), b=wetness, a=conductivity.
in vec2 uv; out vec4 o;
uniform sampler2D uS;
uniform vec2 uPx;
uniform float uStep, uDt;
const vec2 D8[8]=vec2[8](vec2(1,0),vec2(-1,0),vec2(0,1),vec2(0,-1),vec2(.707,.707),vec2(-.707,.707),vec2(.707,-.707),vec2(-.707,-.707));
void main(){
  vec4 c=texture(uS,uv);
  float fi=0., fw=0.;
  for(int i=0;i<8;i++){
    vec4 n=texture(uS,uv+D8[i]*uPx*uStep);
    float kf=sqrt(max(c.a*n.a,0.));
    float g=kf*smoothstep(.02,.3,min(max(c.b,n.b),1.));
    fi+=g*(n.r-c.r);
    fw+=kf*(n.b-c.b);
  }
  c.r=max(c.r+uDt*fi,0.);
  c.b=max(c.b+uDt*fw*.8,0.);
  o=c;
}
